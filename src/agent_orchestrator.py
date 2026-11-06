"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════

# Return windows (days) per customer tier - applied by the RefundAgent.
RETURN_WINDOW_DAYS = {'Standard': 30, 'Premium': 60}


def _to_plain(value):
    """Convert DynamoDB values (Decimal, nested maps/lists) to JSON-friendly types."""
    from decimal import Decimal
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _today() -> str:
    """Today's date (UTC) as YYYY-MM-DD - lets agents compute order age."""
    return time.strftime('%Y-%m-%d', time.gmtime())


def _days_since(date_str: str) -> Optional[int]:
    """Whole days between an ISO date (YYYY-MM-DD) and today, or None if unparsable."""
    from datetime import date
    try:
        return (date.fromisoformat(_today()) - date.fromisoformat(date_str)).days
    except (TypeError, ValueError):
        return None


def _check_return_eligibility(order: dict, tier: Optional[str]) -> dict:
    """
    Apply the return policy in code: the order must be "delivered" and its age
    (days since order_date) must be within the tier's window (Standard 30 /
    Premium 60). The model explains the decision; this function makes it.

    Returns:
        {'eligible': bool, 'reason': str, 'tier': str, 'window_days': int,
         'days_since_order': int | None, 'status': str}
    """
    tier        = tier if tier in RETURN_WINDOW_DAYS else 'Standard'
    window_days = RETURN_WINDOW_DAYS[tier]
    status      = order.get('status', 'unknown')
    days        = _days_since(order.get('order_date'))
    result = {'tier': tier, 'window_days': window_days,
              'days_since_order': days, 'status': status}

    if status == 'return_initiated':
        reason = (f"A return already exists for this order "
                  f"(reference {order.get('return_reference', 'on file')}).")
    elif status != 'delivered':
        reason = f"Only delivered orders can be returned; this order is '{status}'."
    elif days is None:
        reason = "The order date is missing, so the return window cannot be checked."
    elif days > window_days:
        reason = (f"The order is {days} days old, outside the {window_days}-day "
                  f"return window for {tier} customers.")
    else:
        result.update(eligible=True,
                      reason=(f"The order is {days} days old, within the {window_days}-day "
                              f"return window for {tier} customers."))
        return result
    result.update(eligible=False, reason=reason)
    return result


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """You are the InventoryAgent for NovaMart customer support.
Your ONLY job is to gather accurate facts from NovaMart's order and customer records.

Tools:
- check_order_status(customer_id, order_id): one specific order
- get_customer_tier(customer_id): the customer's tier (Standard / Premium) and profile
- list_customer_orders(customer_id): every order the customer has placed

How to work:
1. Read the request and the customer ID you are given.
2. For any order, return or refund request: call get_customer_tier AND check_order_status
   for the order mentioned. If no order ID is given, call list_customer_orders and
   identify the order that matches the product the customer describes.
3. For account questions ("what is my tier?", "am I premium?"): call get_customer_tier.
4. For order history questions: call list_customer_orders.

Report a concise factual summary: customer name and tier, and for each relevant order
its order_id, product, status, order_date, days_since_order, price and quantity.
If a record is not found, say so plainly.

STRICT RULES:
- Report facts only. NEVER decide or suggest whether a return or refund is eligible,
  approved or denied - the RefundAgent makes that decision.
- Never invent data that the tools did not return."""

    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            plus days_since_order and today's date, or a not-found message
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        item = table.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not item:
            return {'found': False, 'customer_id': customer_id, 'order_id': order_id,
                    'message': f"No order {order_id} found for customer {customer_id}."}
        order = _to_plain(item)
        order['found'] = True
        order['today'] = _today()
        order['days_since_order'] = _days_since(order.get('order_date'))
        return order

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details, or a not-found message
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        item = table.get_item(Key={'customer_id': customer_id}).get('Item')
        if not item:
            return {'found': False, 'customer_id': customer_id,
                    'message': f"No customer record found for {customer_id}."}
        profile = _to_plain(item)
        profile['found'] = True
        profile['return_window_days'] = RETURN_WINDOW_DAYS.get(profile.get('tier'), 30)
        return profile

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Dict with customer_id, order_count and a list of orders, each with
            order_id, product_name, status, order_date, days_since_order, price and quantity
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.query(KeyConditionExpression=Key('customer_id').eq(customer_id))
        orders = []
        for item in response.get('Items', []):
            order = _to_plain(item)
            orders.append({
                'order_id':         order.get('order_id'),
                'product_name':     order.get('product_name'),
                'status':           order.get('status'),
                'order_date':       order.get('order_date'),
                'days_since_order': _days_since(order.get('order_date')),
                'price':            order.get('price'),
                'quantity':         order.get('quantity'),
            })
        orders.sort(key=lambda o: o.get('order_date') or '', reverse=True)
        return {'customer_id': customer_id, 'today': _today(),
                'order_count': len(orders), 'orders': orders}

    return Agent(
        name='InventoryAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = f"""You are the RefundAgent for NovaMart customer support.
You decide whether a return/refund is eligible and, when it is, initiate it.

Return windows by customer tier (measured from the order_date, i.e. days_since_order):
- Standard customers: {RETURN_WINDOW_DAYS['Standard']} days
- Premium customers:  {RETURN_WINDOW_DAYS['Premium']} days

Decision process - follow it in order, every time:
1. ALWAYS call get_inventory_context(session_id) FIRST to read the facts the
   InventoryAgent gathered (customer tier, order status, order_date, days_since_order).
   Never decide without these facts. If they are missing, say the decision cannot be
   made yet and stop.
2. The policy: an order is eligible only if its status is "delivered" (processing or
   shipped orders have not arrived; cancelled orders have nothing to return;
   "return_initiated" means a return already exists) AND days_since_order is within
   the tier's window.
3. If the customer ASKED FOR a return/refund: call initiate_refund(customer_id,
   order_id, reason) exactly once - do not pre-judge eligibility yourself. The tool
   applies the policy in code and is the final authority: success=true means
   APPROVED; success=false means DENIED with the tool's "reason". Never retry it and
   never contradict it.
4. If the customer only asked about status or eligibility, do NOT call
   initiate_refund; explain how the policy applies to the facts (INFO ONLY).
5. Report the decision in this format:
   DECISION: APPROVED | DENIED | INFO ONLY
   Order: <order_id> - <product>
   Customer tier: <tier> (<window>-day window), order age: <days> days
   Reason: <one or two sentences>
   Return reference: <reference, if initiated>
   Next steps: <instructions returned by initiate_refund, if any>

Never invent order data or reference numbers."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            Dict with session_id, customer_id, today's date and the inventory_agent
            findings from WorkflowState, or an empty dict if not yet set
        """
        state = _read_workflow_state(session_id)
        if not state or not state.get('inventory_agent'):
            return {}
        return {
            'session_id':      session_id,
            'customer_id':     state.get('customer_id'),
            'today':           _today(),
            'inventory_agent': state['inventory_agent'],
        }

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        The return policy is enforced here, in code: the tool re-reads the order
        and the customer's tier, and only creates the return if the order is
        "delivered" and within the tier's window (Standard 30 / Premium 60 days).
        The write is conditional on status = "delivered", so an expired order can
        never be returned and the same order can never be returned twice.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            On success: success=True, return_reference, instructions and the
            eligibility details. Otherwise: success=False with the reason
            (order not found, not delivered, outside the window, already returned).
        """
        orders = dynamodb.Table(config.ORDERS_TABLE)
        order = orders.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not order:
            return {'success': False,
                    'reason': f"Order {order_id} not found for customer {customer_id}."}
        customer = dynamodb.Table(config.CUSTOMERS_TABLE).get_item(
            Key={'customer_id': customer_id}).get('Item') or {}

        eligibility = _check_return_eligibility(_to_plain(order), customer.get('tier'))
        if not eligibility['eligible']:
            return {'success': False, 'order_id': order_id, **eligibility}

        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        try:
            orders.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression=('SET #s = :status, return_reference = :ref, '
                                  'return_reason = :reason, return_initiated_at = :ts'),
                # Guards against a concurrent second return of the same order.
                ConditionExpression='#s = :delivered',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={
                    ':status':    'return_initiated',
                    ':delivered': 'delivered',
                    ':ref':       return_reference,
                    ':reason':    reason or 'not specified',
                    ':ts':        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                },
            )
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return {'success': False, 'order_id': order_id,
                    'reason': "The order is no longer in 'delivered' status - a return "
                              "may already have been initiated."}
        return {
            'success':          True,
            'order_id':         order_id,
            'status':           'return_initiated',
            'return_reference': return_reference,
            'eligibility':      eligibility,
            'instructions': (
                "A prepaid return shipping label will be emailed to the customer. "
                "Pack the item securely in its original packaging with all accessories, "
                "attach the label and drop it off at any authorized carrier location. "
                "The refund is issued to the original payment method within 5-7 business "
                "days of NovaMart receiving the return."
            ),
        }

    return Agent(
        name='RefundAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # Design choice: the retrievers are agents (not plain Retrieve calls) because
    # each one REWRITES the customer's question into a search query for its own
    # domain - a shipping question phrased around a return still pulls the right
    # shipping passages. That reasoning step is what the extra model call buys.
    _DOMAIN_FOCUS = {
        'returns':  'return windows, eligible and ineligible items, the return process, '
                    'refund methods and timelines, exchanges, damaged items',
        'shipping': 'shipping options and rates, delivery times, order processing, '
                    'tracking, delivery issues, international shipping',
        'warranty': 'warranty length and coverage, exclusions, claims, extended '
                    'protection plans',
    }

    def _retriever_agent(name: str, domain: str, retrieval_tool) -> Agent:
        """Create one deterministic retriever sub-agent bound to a single KB tool."""
        return Agent(
            name=name,
            model=BedrockModel(
                model_id=config.WORKER_MODEL_ID,
                region_name=config.AWS_REGION,
                temperature=0.0,
            ),
            system_prompt=(
                f"You are the {name}, a retrieval specialist for NovaMart's {domain} policy "
                f"knowledge base, which covers: {_DOMAIN_FOCUS[domain]}.\n"
                f"1. Rewrite the customer's question into one short search query that targets "
                f"the {domain} aspects of it, using this domain's vocabulary and keeping any "
                f"customer tier (Standard / Premium) or product category it mentions.\n"
                f"2. Call your retrieval tool with that rewritten query. Only if the passages "
                f"are clearly off-topic, call it once more with a different phrasing (at most "
                f"two calls).\n"
                f"3. Reply with the line 'Search query: <query used>' followed by the relevant "
                f"passages quoted faithfully, each with its source URI. Do not answer from your "
                f"own knowledge or add commentary. If nothing relevant is found, say "
                f"'No relevant {domain} policy passages found.'"
            ),
            tools=[retrieval_tool],
            callback_handler=None,
        )

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: Natural-language policy question

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Returns KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.RETURNS_KB_ID, query))

    returns_retriever = _retriever_agent('ReturnsPolicyRetrieverAgent', 'returns',
                                         retrieve_returns_policy)

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: Natural-language policy question

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Shipping KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query))

    shipping_retriever = _retriever_agent('ShippingPolicyRetrieverAgent', 'shipping',
                                          retrieve_shipping_policy)

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: Natural-language policy question

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Warranty KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query))

    warranty_retriever = _retriever_agent('WarrantyPolicyRetrieverAgent', 'warranty',
                                          retrieve_warranty_policy)

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            # Each call is an independent lookup - start from an empty history so
            # the retriever stays deterministic and its context does not grow.
            agent.messages.clear()
            result = agent(f"Retrieve the {domain.lower()} policy passages relevant to: {query}")
            return domain, str(result).strip()

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    _, text = future.result()
                    results[domain] = text or '[No results]'
                except Exception as exc:
                    logger.warning("%s retriever failed: %s", domain, exc)
                    results[domain] = f"[{domain} retrieval failed: {exc}]"

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        return "\n\n".join(
            f"=== {domain} Policy (from {domain}PolicyRetrieverAgent) ===\n"
            f"{results.get(domain, '[No results]')}"
            for domain in ['Returns', 'Shipping', 'Warranty']
        )

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    system_prompt = """You are the PolicyAgent for NovaMart customer support - the coordinator
of a multi-agent RAG system over three policy knowledge bases (Returns, Shipping, Warranty).

How to work:
1. ALWAYS call search_all_policies(query) FIRST, passing the customer's policy question.
   It queries all three knowledge bases in parallel through specialised retriever agents.
2. Synthesize the retrieved passages into a clear, complete answer to the question.

Grounding rules:
- Use ONLY facts stated in the retrieved passages. Never rely on outside knowledge.
- When the question concerns a customer tier (Standard vs Premium), state the
  tier-specific terms (return window, shipping benefits, warranty length).
- Mention which policy area each fact comes from (Returns / Shipping / Warranty).
- If the passages do not answer the question, say so plainly instead of guessing.
- You answer policy questions only; you have no access to individual customer accounts
  or orders."""

    return Agent(
        name='PolicyAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3,
    )

    system_prompt = """You are the CommunicationAgent for NovaMart customer support.
You write the final message the customer will read.

How to work:
1. ALWAYS call get_full_workflow_context(session_id) FIRST to read everything the other
   agents found: inventory_agent (order and account facts), policy_agent (policy answers)
   and refund_agent (return/refund decision).
2. Compose one reply that answers the customer's original request.

Content rules:
- Include every relevant detail from the workflow findings: order ID and product,
  the return/refund decision and its reason, the return reference number, next steps
  and timelines, and the tier-specific policy terms that apply.
- Use only facts present in the workflow context - never invent order details,
  reference numbers or policy terms. If something needed is missing, say what the
  customer can do next instead of guessing.
- If the request is a calculation (e.g. prices, discounts, totals) and no other agent
  ran, work it out yourself step by step: keep full precision through every step and
  round to two decimals only at the end (e.g. 5 x $29.99 = $149.95; 10% off = $134.955,
  which rounds to $134.96). Show the steps briefly.
- Answer only the current request; ignore findings that clearly belong to an earlier,
  unrelated question in the same session.

Tone and format:
- Warm, empathetic and professional. Greet the customer by first name when it is known.
- Clear and concise: short paragraphs or a short list for steps. Plain text only - no
  XML tags, no internal agent names, no mention of tools or workflow state.
- Close by offering further help, signed "NovaMart Customer Support"."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (customer_id, inventory_agent, policy_agent,
            refund_agent, version), or an error message if the session does not exist
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {'error': f"No WorkflowState found for session {session_id}."}
        state = _to_plain(state)
        state.pop('ttl', None)
        return state

    return Agent(
        name='CommunicationAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    from strands.tools.executors import SequentialToolExecutor

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
    )

    system_prompt = """You are the OrchestratorAgent for NovaMart customer support.
You ROUTE customer requests to specialist agents. You never answer the customer yourself.

Every message starts with "[Session ID: <id>] [Customer ID: <id>]" followed by the
customer's request. Pass those exact IDs to every tool. Call tools ONE AT A TIME, in order.

ROUTING RULES (follow exactly):
Rule 1 - Every request, always: call initialize_session FIRST.
Rule 2 - Order status, return or refund requests (e.g. "where is my order?",
         "I want to return ORD-123", "can I get a refund?"): call
         route_to_inventory_agent, THEN route_to_refund_agent.
Rule 3 - Policy meaning questions (return windows, shipping rates/options, warranty
         terms, what a tier includes in general): call route_to_policy_agent.
Rule 4 - Account questions about THIS customer ("what is my tier?", "am I premium?",
         "what orders do I have?"): call route_to_inventory_agent.
         NEVER use route_to_policy_agent for these - it only knows policy text,
         not customer data.
Rule 5 - Math / calculation questions (prices, discounts, totals): do NOT call
         route_to_inventory_agent, route_to_policy_agent or route_to_refund_agent.
         Go straight to route_to_communication_agent, which performs the calculation.
Rule 6 - Every request, always, as the LAST step: call route_to_communication_agent
         with the customer's original request to compose the final reply.

A request may match several rules (e.g. a return plus a policy question): run every
matching specialist in the order above, then finish with Rule 6.

CRITICAL:
- You must NEVER write the customer-facing response yourself, even when you think you
  already know the answer. route_to_communication_agent is ALWAYS your final tool call
  - no exceptions.
- After route_to_communication_agent returns, reply with its text exactly as returned,
  with nothing added, removed or reworded."""

    def _route(column: str, worker: Agent, session_id: str, customer_id: str,
               prompt: str) -> str:
        """
        Shared routing pattern used by every route_to_* tool:
          1. read the current WorkflowState (version = optimistic-lock token)
          2. invoke the worker agent
          3. write its result to the worker's column with
             _update_workflow_state(..., expected_version=<version read in step 1>)
        """
        state = _read_workflow_state(session_id)
        if state is None:                       # initialize_session was skipped
            state = _create_workflow_state(session_id, customer_id or 'UNKNOWN')
        old_version = int(state['version'])

        trace.step_start(column)
        try:
            result_text = _strip_xml_tags(str(worker(prompt))).strip()
        except Exception as exc:
            logger.warning("%s failed: %s", column, exc)
            result_text = f"[{column} could not complete the request: {exc}]"

        _update_workflow_state(session_id, {column: result_text},
                               expected_version=old_version)
        trace.step_done(column, old_version)
        return result_text

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}]\n"
                  f"Customer request: {request}\n"
                  f"Gather the order and account facts needed for this request.")
        return _route('inventory_agent', inventory_agent, session_id, customer_id, prompt)

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id) or {}
        prompt = f"Customer policy question: {request}"
        return _route('policy_agent', policy_agent, session_id,
                      state.get('customer_id', 'UNKNOWN'), prompt)

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}]\n"
                  f"Today's date: {_today()}\n"
                  f"Customer request: {request}\n"
                  f"Read the inventory context for this session and make the "
                  f"return/refund decision.")
        return _route('refund_agent', refund_agent, session_id, customer_id, prompt)

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}]\n"
                  f"Customer's original message: {original_request}\n"
                  f"Read the full workflow context for this session and write the final "
                  f"reply to the customer.")
        return _route('communication_agent', communication_agent, session_id,
                      customer_id, prompt)

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            state = _create_workflow_state(session_id, customer_id)
            return (f"Session {session_id} initialized for customer {customer_id} "
                    f"(WorkflowState version {state['version']}).")
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # Follow-up turn in the same chat session - keep the existing record.
            state = _read_workflow_state(session_id) or {}
            return (f"Session {session_id} already exists for customer "
                    f"{state.get('customer_id', customer_id)} (WorkflowState version "
                    f"{int(state.get('version', 0))}) - continuing the session.")

    return Agent(
        name='OrchestratorAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[initialize_session, route_to_inventory_agent, route_to_policy_agent,
               route_to_refund_agent, route_to_communication_agent],
        # Run tool calls strictly one after another so the routing order
        # (initialize -> workers -> communication) is preserved.
        tool_executor=SequentialToolExecutor(),
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    content_filters = [
        {'type': ftype, 'inputStrength': strength, 'outputStrength': strength}
        for ftype, strength in (('SEXUAL', 'HIGH'), ('VIOLENCE', 'HIGH'), ('HATE', 'HIGH'),
                                ('INSULTS', 'MEDIUM'), ('MISCONDUCT', 'MEDIUM'))
    ]

    pii_entities = [
        {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
        {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
        {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
        {'type': 'PHONE',                     'action': 'ANONYMIZE'},
    ]

    # One DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS
    topic_definitions = {
        'competitor products': {
            'name': 'Competitor_Products',
            'definition': ('Requests to discuss, compare, recommend or endorse products, '
                           'prices or services of other retailers or competing stores '
                           'instead of NovaMart.'),
            'examples': [
                'Is Amazon cheaper than NovaMart for these headphones?',
                'Should I buy this laptop at Best Buy instead?',
                'Which competitor store has better deals than you?',
            ],
        },
        'pricing negotiations': {
            'name': 'Pricing_Negotiations',
            'definition': ('Haggling, bargaining or asking NovaMart to lower, match or '
                           'change an advertised price. Calculating totals from a stated '
                           'price and discount is not a negotiation.'),
            'examples': [
                'Can you give me a better price on this laptop?',
                'I will only buy it if you drop the price by $50.',
                'Will you match the lower price I found elsewhere?',
                "Lower the price and I'll order two.",
            ],
        },
        'legal threats': {
            'name': 'Legal_Threats',
            'definition': ('Threats of lawsuits, legal action, attorneys or court '
                           'proceedings against NovaMart or its staff.'),
            'examples': [
                'I am going to sue NovaMart if you do not refund me.',
                'My lawyer will be contacting you about this order.',
                'I will take you to court over this.',
            ],
        },
    }
    topics = [
        {**topic_definitions[topic], 'type': 'DENY'}
        for topic in config.GUARDRAIL_BLOCKED_TOPICS
    ]

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=('NovaMart customer support guardrail: harmful content, PII, '
                     'off-topic subjects and profanity.'),
        contentPolicyConfig={'filtersConfig': content_filters},
        sensitiveInformationPolicyConfig={'piiEntitiesConfig': pii_entities},
        topicPolicyConfig={
            'topicsConfig': topics,
            'tierConfig':   {'tierName': 'STANDARD'},
        },
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging=(
            "Sorry, I can't help with that request. I'm happy to assist with your "
            "NovaMart orders, returns, refunds, shipping or warranty questions."
        ),
        blockedOutputsMessaging=(
            "Sorry, I'm unable to share that response. Please contact NovaMart "
            "Customer Support if you need further help with your order."
        ),
    )
    guardrail_id = response['guardrailId']
    print(f"  Guardrail created: {guardrail_id} (DRAFT)")

    # Wait until the DRAFT is READY, then publish a numbered version.
    deadline = time.time() + 120
    while time.time() < deadline:
        status = bedrock_client.get_guardrail(guardrailIdentifier=guardrail_id)['status']
        if status == 'READY':
            break
        if status == 'FAILED':
            raise RuntimeError(f"Guardrail {guardrail_id} creation failed")
        time.sleep(3)

    version = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Initial published version for the NovaMart support agents',
    )
    guardrail_version = version['version']
    print(f"  Guardrail version published: {guardrail_version}")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the pre-written staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    runtime_env = {
        'AWS_REGION':        config.AWS_REGION,
        'PROJECT_NAME':      config.PROJECT_NAME,
        'RETURNS_KB_ID':     config.RETURNS_KB_ID,
        'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':      guardrail_id,
        'GUARDRAIL_VERSION': guardrail_version,
    }
    missing = [k for k, v in runtime_env.items() if not v]
    if missing:
        print(f"  [Warning] Runtime environment variables without a value: {missing}")

    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )
    agentcore_cli.deploy()
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError(
            "agentcore deploy finished but no runtime ARN was recorded in "
            "agentcore/.cli/deployed-state.json - check the CLI output above "
            "(`agentcore status`, agentcore/.cli/logs/deploy/)")

    # Wait for the runtime to become READY and return its ARN.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description=('NovaMart customer support - session-scoped conversation summaries '
                     'so customers do not have to repeat themselves between turns.'),
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name':        'SessionSummary',
                'description': 'Summarizes each customer support session',
                'namespaces':  ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,
        },
    }

    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        xray = summary.get('xray', {})
        print(f"  CloudWatch Logs : {summary['log_group']} "
              f"(level {logging_configuration['cloudWatchConfig']['logLevel']})")
        print(f"  X-Ray tracing   : sampling rate "
              f"{logging_configuration['xRayConfig']['samplingRate']:.0%} "
              f"(Transaction Search -> {xray.get('destination')}, "
              f"indexing {xray.get('indexing_percent')}%)")
    except Exception as e:
        print(f"  [Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


# 'novamart.*' loggers are shipped to CloudWatch by setup_logging().
_run_logger = logging.getLogger('novamart.orchestrator')


def _finalize_turn(session_id: str, response, previous_reply: str = '',
                   trace_root=None) -> tuple[str, bool]:
    """
    Resolve the customer-facing reply for one turn and verify the routing finished.

    The agent loop stopping (end_turn) only means the model stopped talking. A turn
    is complete only if the CommunicationAgent wrote a NEW reply to WorkflowState
    during this turn (compared with `previous_reply`, the value before the turn -
    chat sessions keep earlier replies). Incomplete turns are logged as a WARNING
    (CloudWatch) and annotated routing_complete=false on the X-Ray trace, so they
    can be counted instead of looking like a success.

    Call it inside the tracer.trace_request() block so the annotation is published.

    Returns:
        (reply_text, routing_complete)
    """
    state    = _read_workflow_state(session_id) or {}
    reply    = state.get('communication_agent', '') or ''
    complete = bool(reply) and reply != previous_reply
    if trace_root is not None:
        trace_root.annotations['routing_complete'] = complete
    if complete:
        _run_logger.info("run complete | session=%s version=%s",
                         session_id, state.get('version'))
        return _strip_xml_tags(reply), True
    _run_logger.warning(
        "INCOMPLETE run | session=%s - CommunicationAgent did not write a reply this turn; "
        "returning the orchestrator's own text instead", session_id)
    return _strip_xml_tags(str(response)), False


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query) as root:
            response = orchestrator(prompt)
            _, complete = _finalize_turn(session_id, response, trace_root=root)
        print(f"Response: {response}")
        if not complete:
            print("  [WARNING] Routing incomplete - the CommunicationAgent did not run.")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        previous_reply = (_read_workflow_state(session_id) or {}).get('communication_agent', '')

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input) as root:
                response = orchestrator(prompt)
                # ── Resolve the final customer-facing text ────────────
                text, complete = _finalize_turn(session_id, response,
                                                previous_reply, trace_root=root)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if not complete:
            print(f"  {_C.ERR}[WARNING] Routing incomplete - "
                  f"the CommunicationAgent did not run this turn.{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        previous_reply = (_read_workflow_state(session_id) or {}).get('communication_agent', '')
        with tracer.trace_request(session_id, customer_id, prompt) as root:
            response = _orchestrator()(enriched)
            text, complete = _finalize_turn(session_id, response, previous_reply,
                                            trace_root=root)
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'routing_complete': complete, 'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
