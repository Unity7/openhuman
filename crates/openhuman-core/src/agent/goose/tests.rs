use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::json;
use tinyagents_harness::host::security_gate::{GateDecision, ToolCallRequest};
use tinyinference::{
    message::AssistantMessage,
    model::{ChatModel, ModelRequest, ModelResponse},
    tool::ToolCall,
    usage::Usage,
};
use tokio_util::sync::CancellationToken;

use crate::{
    agent::{
        messages::{ChatMessage, ConversationMessage},
        primary_orchestration::capability::{
            CapabilityAvailability, CapabilityBackend, CapabilityModality, CapabilityOperation,
            CapabilitySideEffect, MonetaryBoundary, ToolCapability, ToolRoute,
        },
        progress::AgentProgress,
    },
    tools::{Tool, ToolResult},
};

use goose_provider_types::conversation::message::MessageContent;

use super::{
    convert::{final_text, goose_to_openhuman},
    store::FileGooseCheckpointStore,
    GooseCheckpointStore, GooseStopReason, GooseToolSecurity, GooseTurnAdapter,
    InMemoryGooseCheckpointStore,
};

struct ScriptedModel {
    responses: Mutex<VecDeque<ModelResponse>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ScriptedModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedModel {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request);
        self.responses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop_front()
            .ok_or_else(|| tinyinference::Error::Model("script exhausted".into()))
    }
}

struct CountingTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        "read_counter"
    }

    fn description(&self) -> &str {
        "Returns a deterministic counter value without external I/O."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(format!("value:{}", args["key"])))
    }
}

struct NamedCountingTool {
    name: String,
    calls: Arc<AtomicUsize>,
}

impl NamedCountingTool {
    fn new(name: impl Into<String>, calls: Arc<AtomicUsize>) -> Self {
        Self {
            name: name.into(),
            calls,
        }
    }
}

#[async_trait]
impl Tool for NamedCountingTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Returns a deterministic counter value without external I/O."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let val = args.get("key").and_then(|k| k.as_str()).unwrap_or("ok");
        Ok(ToolResult::success(format!("{}:{}", self.name, val)))
    }
}

#[derive(Default)]
struct AllowSecurity;

#[async_trait]
impl GooseToolSecurity for AllowSecurity {
    async fn authorize(&self, _request: &ToolCallRequest) -> Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    fn record_execution(&self, _call_id: &str, _success: bool, _error: Option<&str>) {}
}

fn tool_response(usage: Usage) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some("assistant-tool".into()),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(
                "call-1",
                "read_counter",
                json!({"key": "alpha"}),
            )],
            usage: Some(usage),
        },
        usage: Some(usage),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: Some("tool".into()),
        served_from_cache: false,
    }
}

fn final_response(text: &str, usage: Usage) -> ModelResponse {
    ModelResponse::assistant(text).with_usage(usage)
}

fn named_tool_response(
    call_id: &str,
    tool_name: &str,
    key_val: &str,
    usage: Usage,
) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("assistant-{call_id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(call_id, tool_name, json!({"key": key_val}))],
            usage: Some(usage),
        },
        usage: Some(usage),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: Some("tool".into()),
        served_from_cache: false,
    }
}

fn test_route(name: &str, priority: u16, registration_index: usize) -> ToolRoute {
    ToolRoute::new(ToolCapability {
        name: name.to_string(),
        operations: vec![CapabilityOperation::ReadWorkspace],
        modalities: vec![CapabilityModality::Text],
        backend: CapabilityBackend::Local,
        monetary_boundary: MonetaryBoundary::NonMetered,
        side_effect: CapabilitySideEffect::LocalRead,
        availability: CapabilityAvailability::Available,
        permission: Default::default(),
        priority,
        registration_index,
    })
}

fn default_route(name: &str) -> ToolRoute {
    test_route(name, 100, 0)
}

fn adapter(
    store: Arc<dyn GooseCheckpointStore>,
    model: Arc<dyn ChatModel<()>>,
    calls: Arc<AtomicUsize>,
    security: Arc<dyn GooseToolSecurity>,
    progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
    cancel: CancellationToken,
) -> GooseTurnAdapter {
    adapter_with_snapshots_and_routes(
        store,
        model,
        Arc::new(vec![Box::new(CountingTool { calls })]),
        Arc::new(Vec::new()),
        vec![default_route("read_counter")],
        security,
        progress,
        cancel,
    )
}

fn adapter_with_snapshots_and_routes(
    store: Arc<dyn GooseCheckpointStore>,
    model: Arc<dyn ChatModel<()>>,
    durable_tools: Arc<Vec<Box<dyn Tool>>>,
    synthesized_tools: Arc<Vec<Box<dyn Tool>>>,
    routes: Vec<ToolRoute>,
    security: Arc<dyn GooseToolSecurity>,
    progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
    cancel: CancellationToken,
) -> GooseTurnAdapter {
    GooseTurnAdapter {
        store,
        model,
        model_name: "mock-primary".into(),
        provider_id: "mock".into(),
        durable_tools,
        synthesized_tools,
        routes,
        security,
        progress,
        cancel,
        max_output_tokens: Some(64),
        max_primary_calls: 12,
    }
}

fn kickoff() -> Vec<ConversationMessage> {
    vec![ConversationMessage::Chat(ChatMessage::user("read alpha"))]
}

fn to_qwen_route(mut adapter: GooseTurnAdapter) -> GooseTurnAdapter {
    adapter.provider_id = "lmstudio".into();
    adapter.model_name = "qwen38-openhuman".into();
    adapter
}

#[test]
fn openhuman_message_shapes_round_trip_through_goose() {
    let source = vec![
        ConversationMessage::Chat(ChatMessage::system("system")),
        ConversationMessage::Chat(ChatMessage::user("hello")),
        ConversationMessage::AssistantToolCalls {
            text: Some("checking".into()),
            tool_calls: vec![crate::inference::provider::ToolCall {
                id: "call-1".into(),
                name: "read_counter".into(),
                arguments: json!({"key": "alpha"}).to_string(),
                extra_content: None,
            }],
            reasoning_content: Some("private".into()),
            extra_metadata: Some(json!({"source": "fixture"})),
        },
        ConversationMessage::ToolResults(vec![crate::agent::messages::ToolResultMessage {
            tool_call_id: "call-1".into(),
            content: "value:alpha".into(),
        }]),
        ConversationMessage::Chat(ChatMessage::assistant("done")),
    ];
    let checkpoint = GooseTurnAdapter::checkpoint_from_openhuman(&source).unwrap();
    let roundtrip = goose_to_openhuman(&checkpoint.conversation);
    assert_eq!(
        serde_json::to_value(roundtrip).unwrap(),
        serde_json::to_value(source).unwrap()
    );
    assert!(checkpoint.actions["call-1"].observation.is_some());
}

#[tokio::test]
async fn state_machine_persists_action_before_execution_and_one_observation() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "turn",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response(Usage::new(10, 2)),
        final_response("finished", Usage::new(20, 3)),
    ]));
    let calls = Arc::new(AtomicUsize::new(0));
    let outcome = adapter(
        store.clone(),
        model.clone(),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("turn")
    .await
    .unwrap();

    assert_eq!(outcome.stop_reason, GooseStopReason::FinalAnswer);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(outcome.checkpoint.actions.len(), 1);
    assert_eq!(
        outcome.checkpoint.actions["call-1"]
            .observation
            .as_ref()
            .unwrap()
            .output,
        "value:\"alpha\""
    );
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    assert_eq!(
        model.requests.lock().unwrap()[0].tools[0].name,
        "read_counter"
    );

    let resumed = adapter(
        store,
        model,
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("turn")
    .await
    .unwrap();
    assert_eq!(resumed.stop_reason, GooseStopReason::FinalAnswer);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn primary_call_ceiling_yields_a_resumable_checkpoint() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "ceiling",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let model = Arc::new(ScriptedModel::new(vec![tool_response(Usage::new(10, 2))]));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut bounded = adapter(
        store,
        model.clone(),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    );
    bounded.max_primary_calls = 1;

    let outcome = bounded.run("ceiling").await.unwrap();

    assert_eq!(outcome.stop_reason, GooseStopReason::CallCeiling);
    assert_eq!(outcome.checkpoint.usage.primary_calls, 1);
    assert!(outcome.checkpoint.is_resumable());
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct DenySecurity;

#[async_trait]
impl GooseToolSecurity for DenySecurity {
    async fn authorize(&self, _request: &ToolCallRequest) -> Result<GateDecision> {
        Ok(GateDecision::deny("fixture policy denied this call"))
    }

    fn record_execution(&self, _call_id: &str, _success: bool, _error: Option<&str>) {}
}

#[tokio::test]
async fn authorization_denial_is_a_single_observation_and_never_executes() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "denied",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let outcome = adapter(
        store,
        Arc::new(ScriptedModel::new(vec![
            tool_response(Usage::new(10, 2)),
            final_response("cannot run that", Usage::new(20, 3)),
        ])),
        calls.clone(),
        Arc::new(DenySecurity),
        None,
        CancellationToken::new(),
    )
    .run("denied")
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let observation = outcome.checkpoint.actions["call-1"]
        .observation
        .as_ref()
        .unwrap();
    assert!(!observation.success);
    assert_eq!(observation.output, "fixture policy denied this call");
}

#[tokio::test]
async fn latest_context_occupancy_is_not_cumulative_traffic() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "usage",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response(Usage::new(100, 5)),
        final_response("finished", Usage::new(140, 7)),
    ]));
    let outcome = adapter(
        store,
        model,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("usage")
    .await
    .unwrap();
    assert_eq!(outcome.checkpoint.usage.primary_calls, 2);
    assert_eq!(outcome.checkpoint.usage.latest_primary_input_tokens, 140);
    assert_eq!(outcome.checkpoint.usage.cumulative_input_tokens, 240);
    assert_eq!(outcome.checkpoint.usage.cumulative_output_tokens, 12);
}

struct WaitingSecurity {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl GooseToolSecurity for WaitingSecurity {
    async fn authorize(&self, _request: &ToolCallRequest) -> Result<GateDecision> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(GateDecision::Prompted { approved: true })
    }

    fn record_execution(&self, _call_id: &str, _success: bool, _error: Option<&str>) {}
}

#[tokio::test]
async fn approval_wait_occurs_after_action_checkpoint_and_before_execution() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "approval",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let security = Arc::new(WaitingSecurity {
        entered: entered.clone(),
        release: release.clone(),
    });
    let approval_adapter = adapter(
        store.clone(),
        Arc::new(ScriptedModel::new(vec![
            tool_response(Usage::new(10, 2)),
            final_response("approved", Usage::new(20, 2)),
        ])),
        calls.clone(),
        security,
        None,
        CancellationToken::new(),
    );
    let turn = tokio::spawn(async move { approval_adapter.run("approval").await });
    entered.notified().await;
    let waiting = store.load("approval").await.unwrap();
    assert!(waiting.actions.contains_key("call-1"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    release.notify_one();
    assert_eq!(
        turn.await.unwrap().unwrap().stop_reason,
        GooseStopReason::FinalAnswer
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_persists_one_matching_observation_without_execution() {
    let source = vec![
        ConversationMessage::Chat(ChatMessage::user("read alpha")),
        ConversationMessage::AssistantToolCalls {
            text: None,
            tool_calls: vec![crate::inference::provider::ToolCall {
                id: "call-1".into(),
                name: "read_counter".into(),
                arguments: json!({"key": "alpha"}).to_string(),
                extra_content: None,
            }],
            reasoning_content: None,
            extra_metadata: None,
        },
    ];
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "cancel",
        GooseTurnAdapter::checkpoint_from_openhuman(&source).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = adapter(
        store,
        Arc::new(ScriptedModel::new(Vec::new())),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        cancel,
    )
    .run("cancel")
    .await
    .unwrap();
    assert_eq!(outcome.stop_reason, GooseStopReason::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let observation = outcome.checkpoint.actions["call-1"]
        .observation
        .as_ref()
        .unwrap();
    assert!(!observation.success);
    assert!(observation.output.contains("cancelled"));
}

struct RejectingStore {
    inner: InMemoryGooseCheckpointStore,
}

#[async_trait]
impl GooseCheckpointStore for RejectingStore {
    async fn load(&self, session_id: &str) -> Result<super::GooseCheckpoint> {
        self.inner.load(session_id).await
    }

    async fn compare_and_swap(
        &self,
        _session_id: &str,
        _expected_revision: u64,
        _checkpoint: super::GooseCheckpoint,
    ) -> Result<()> {
        Err(anyhow!("simulated persistence failure"))
    }

    async fn claim_execution(&self, session_id: &str, call_id: &str) -> Result<bool> {
        self.inner.claim_execution(session_id, call_id).await
    }
}

#[tokio::test]
async fn failed_action_checkpoint_prevents_tool_execution() {
    let store = Arc::new(RejectingStore {
        inner: InMemoryGooseCheckpointStore::default(),
    });
    store.inner.insert(
        "fail",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let error = adapter(
        store,
        Arc::new(ScriptedModel::new(vec![tool_response(Usage::new(10, 2))])),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("fail")
    .await
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("persist Goose state-machine step"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct FailObservationOnceStore {
    inner: InMemoryGooseCheckpointStore,
    failed: AtomicBool,
}

#[async_trait]
impl GooseCheckpointStore for FailObservationOnceStore {
    async fn load(&self, session_id: &str) -> Result<super::GooseCheckpoint> {
        self.inner.load(session_id).await
    }

    async fn compare_and_swap(
        &self,
        session_id: &str,
        expected_revision: u64,
        checkpoint: super::GooseCheckpoint,
    ) -> Result<()> {
        let contains_observation = checkpoint
            .actions
            .values()
            .any(|action| action.observation.is_some());
        if contains_observation && !self.failed.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("simulated observation commit failure"));
        }
        self.inner
            .compare_and_swap(session_id, expected_revision, checkpoint)
            .await
    }

    async fn claim_execution(&self, session_id: &str, call_id: &str) -> Result<bool> {
        self.inner.claim_execution(session_id, call_id).await
    }
}

#[tokio::test]
async fn resume_after_observation_commit_failure_refuses_duplicate_side_effect() {
    let store = Arc::new(FailObservationOnceStore {
        inner: InMemoryGooseCheckpointStore::default(),
        failed: AtomicBool::new(false),
    });
    store.inner.insert(
        "crash-window",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response(Usage::new(10, 2)),
        final_response("reconciled", Usage::new(20, 3)),
    ]));
    let calls = Arc::new(AtomicUsize::new(0));
    let first = adapter(
        store.clone(),
        model.clone(),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("crash-window")
    .await;
    assert!(first.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let resumed = adapter(
        store,
        model,
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("crash-window")
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let observation = resumed.checkpoint.actions["call-1"]
        .observation
        .as_ref()
        .unwrap();
    assert!(!observation.success);
    assert!(observation.output.contains("refusing to repeat"));
}

#[tokio::test]
async fn progress_projects_model_tool_usage_and_turn_lifecycle() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "progress",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    adapter(
        store,
        Arc::new(ScriptedModel::new(vec![
            tool_response(Usage::new(10, 2)),
            final_response("done", Usage::new(20, 3)),
        ])),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AllowSecurity),
        Some(tx),
        CancellationToken::new(),
    )
    .run("progress")
    .await
    .unwrap();
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    assert!(matches!(events.first(), Some(AgentProgress::TurnStarted)));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentProgress::ModelCallCompleted { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentProgress::ToolCallStarted { .. }))
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(AgentProgress::TurnCompleted { iterations: 2 })
    ));
}

#[tokio::test]
async fn first_model_request_advertises_tools_in_route_order_even_when_snapshot_order_differs() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "route-order",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls_first = Arc::new(AtomicUsize::new(0));
    let calls_second = Arc::new(AtomicUsize::new(0));

    // Snapshot order has "tool_second" first, then "tool_first".
    let durable_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![
        Box::new(NamedCountingTool::new("tool_second", calls_second.clone())),
        Box::new(NamedCountingTool::new("tool_first", calls_first.clone())),
    ]);
    let synthesized_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(Vec::new());

    // Route order explicitly places "tool_first" before "tool_second".
    let routes = vec![
        test_route("tool_first", 10, 0),
        test_route("tool_second", 20, 1),
    ];

    let model = Arc::new(ScriptedModel::new(vec![final_response(
        "ok",
        Usage::new(10, 2),
    )]));

    let outcome = adapter_with_snapshots_and_routes(
        store,
        model.clone(),
        durable_tools,
        synthesized_tools,
        routes,
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    )
    .run("route-order")
    .await
    .unwrap();

    assert_eq!(outcome.stop_reason, GooseStopReason::FinalAnswer);
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let advertised = &requests[0].tools;
    assert_eq!(advertised.len(), 2);
    assert_eq!(advertised[0].name, "tool_first");
    assert_eq!(advertised[1].name, "tool_second");
}

#[tokio::test]
async fn tool_present_in_either_snapshot_but_omitted_from_routes_is_neither_advertised_nor_executable(
) {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "omitted-durable",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls_durable = Arc::new(AtomicUsize::new(0));
    let calls_synth = Arc::new(AtomicUsize::new(0));
    let calls_routed = Arc::new(AtomicUsize::new(0));

    // "omitted_durable" in durable snapshot, "omitted_synth" in synthesized snapshot.
    let durable_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![
        Box::new(NamedCountingTool::new(
            "omitted_durable",
            calls_durable.clone(),
        )),
        Box::new(NamedCountingTool::new("routed_tool", calls_routed.clone())),
    ]);
    let synthesized_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(
        NamedCountingTool::new("omitted_synth", calls_synth.clone()),
    )]);

    // Only "routed_tool" is in routes.
    let routes = vec![test_route("routed_tool", 10, 0)];

    let model = Arc::new(ScriptedModel::new(vec![named_tool_response(
        "call-omitted",
        "omitted_durable",
        "val",
        Usage::new(10, 2),
    )]));

    let adapter = adapter_with_snapshots_and_routes(
        store.clone(),
        model.clone(),
        durable_tools.clone(),
        synthesized_tools.clone(),
        routes.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    );

    let outcome = adapter.run("omitted-durable").await.unwrap();
    assert_eq!(outcome.stop_reason, GooseStopReason::Yielded);
    assert!(outcome.checkpoint.actions.contains_key("call-omitted"));
    let action = &outcome.checkpoint.actions["call-omitted"];
    assert!(action.observation.is_none());
    assert_eq!(calls_durable.load(Ordering::SeqCst), 0);
    assert_eq!(calls_synth.load(Ordering::SeqCst), 0);
    assert_eq!(calls_routed.load(Ordering::SeqCst), 0);

    // Verify advertising: neither omitted durable nor omitted synth tool is advertised.
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "routed_tool");
    // Also verify omitted synthesized tool execution fails.
    store.insert(
        "omitted-synth",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let model_synth = Arc::new(ScriptedModel::new(vec![named_tool_response(
        "call-synth",
        "omitted_synth",
        "val",
        Usage::new(10, 2),
    )]));
    let adapter_synth = adapter_with_snapshots_and_routes(
        store,
        model_synth,
        durable_tools,
        synthesized_tools,
        routes,
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    );
    let outcome_synth = adapter_synth.run("omitted-synth").await.unwrap();
    assert_eq!(outcome_synth.stop_reason, GooseStopReason::Yielded);
    assert!(outcome_synth.checkpoint.actions.contains_key("call-synth"));
    let action_synth = &outcome_synth.checkpoint.actions["call-synth"];
    assert!(action_synth.observation.is_none());
    assert_eq!(calls_synth.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejected_unplanned_call_reaches_neither_security_authorization_nor_execution() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "unplanned",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let auth_calls = Arc::new(AtomicUsize::new(0));

    struct SpySecurity {
        auth_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl GooseToolSecurity for SpySecurity {
        async fn authorize(&self, _request: &ToolCallRequest) -> Result<GateDecision> {
            self.auth_calls.fetch_add(1, Ordering::SeqCst);
            Ok(GateDecision::Allow)
        }

        fn record_execution(&self, _call_id: &str, _success: bool, _error: Option<&str>) {}
    }

    let durable_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(NamedCountingTool::new(
        "unplanned_tool",
        calls.clone(),
    ))]);
    let synthesized_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(Vec::new());
    // Empty routes: "unplanned_tool" is unplanned.
    let routes = Vec::new();

    let model = Arc::new(ScriptedModel::new(vec![named_tool_response(
        "call-unplanned",
        "unplanned_tool",
        "val",
        Usage::new(10, 2),
    )]));

    let adapter = adapter_with_snapshots_and_routes(
        store,
        model,
        durable_tools,
        synthesized_tools,
        routes,
        Arc::new(SpySecurity {
            auth_calls: auth_calls.clone(),
        }),
        None,
        CancellationToken::new(),
    );

    let outcome = adapter.run("unplanned").await.unwrap();
    assert_eq!(outcome.stop_reason, GooseStopReason::Yielded);
    assert!(outcome.checkpoint.actions.contains_key("call-unplanned"));
    let action = &outcome.checkpoint.actions["call-unplanned"];
    assert!(action.observation.is_none());
    assert_eq!(auth_calls.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn denied_planned_route_cannot_execute() {
    let store = Arc::new(InMemoryGooseCheckpointStore::default());
    store.insert(
        "denied-planned",
        GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let auth_calls = Arc::new(AtomicUsize::new(0));

    struct DenyingSpySecurity {
        auth_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl GooseToolSecurity for DenyingSpySecurity {
        async fn authorize(&self, _request: &ToolCallRequest) -> Result<GateDecision> {
            self.auth_calls.fetch_add(1, Ordering::SeqCst);
            Ok(GateDecision::deny("planned route was denied by policy"))
        }

        fn record_execution(&self, _call_id: &str, _success: bool, _error: Option<&str>) {}
    }

    // Tool is planned in routes.
    let durable_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(NamedCountingTool::new(
        "planned_tool",
        calls.clone(),
    ))]);
    let synthesized_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(Vec::new());
    let routes = vec![test_route("planned_tool", 10, 0)];

    let model = Arc::new(ScriptedModel::new(vec![
        named_tool_response("call-planned", "planned_tool", "val", Usage::new(10, 2)),
        final_response("cannot run that", Usage::new(20, 3)),
    ]));

    let outcome = adapter_with_snapshots_and_routes(
        store,
        model,
        durable_tools,
        synthesized_tools,
        routes,
        Arc::new(DenyingSpySecurity {
            auth_calls: auth_calls.clone(),
        }),
        None,
        CancellationToken::new(),
    )
    .run("denied-planned")
    .await
    .unwrap();

    // Authorization was reached for the planned route.
    assert_eq!(auth_calls.load(Ordering::SeqCst), 1);
    // Tool execution was refused.
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    // Denial observation was durably recorded.
    let observation = outcome.checkpoint.actions["call-planned"]
        .observation
        .as_ref()
        .unwrap();
    assert!(!observation.success);
    assert_eq!(observation.output, "planned route was denied by policy");
}

#[tokio::test]
async fn qwen_correction_persistence_and_terminal_resume_across_store_handles() {
    let dir = tempfile::tempdir().unwrap();
    let session_id = "qwen-correction-session";
    let store1 = Arc::new(FileGooseCheckpointStore::new(dir.path()));
    store1
        .insert(
            session_id,
            GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
        )
        .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let model1 = Arc::new(ScriptedModel::new(vec![final_response(
        "<tool_call>\n{\"name\": \"read_counter\", \"arguments\": {invalid}}\n</tool_call>",
        Usage::new(10, 2),
    )]));
    let mut adapter1 = to_qwen_route(adapter(
        store1.clone(),
        model1,
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    ));
    adapter1.max_primary_calls = 1;

    let outcome1 = adapter1.run(session_id).await.unwrap();

    assert_eq!(outcome1.stop_reason, GooseStopReason::CallCeiling);
    let checkpoint1 = store1.load(session_id).await.unwrap();
    assert_eq!(checkpoint1.protocol_correction_count, 1);
    assert!(!checkpoint1.terminal_protocol_failure);

    let corrections1: Vec<_> = checkpoint1
        .conversation
        .messages()
        .iter()
        .filter(|m| m.metadata.agent_visible && !m.metadata.user_visible)
        .collect();
    assert_eq!(corrections1.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(checkpoint1.actions.is_empty());

    let oh1 = goose_to_openhuman(&checkpoint1.conversation);
    let oh1_str = serde_json::to_string(&oh1).unwrap();
    assert!(!oh1_str.contains("<tool_call>"));
    assert!(!oh1_str.contains("</tool_call>"));
    let outcome1_str = serde_json::to_string(&outcome1.openhuman_messages).unwrap();
    assert!(!outcome1_str.contains("<tool_call>"));
    assert!(!outcome1_str.contains("</tool_call>"));

    let store2 = Arc::new(FileGooseCheckpointStore::new(dir.path()));
    let model2 = Arc::new(ScriptedModel::new(vec![final_response(
        "<tool_call>\n{\"name\": \"read_counter\", \"arguments\": {invalid_again}}\n</tool_call>",
        Usage::new(10, 2),
    )]));
    let adapter2 = to_qwen_route(adapter(
        store2.clone(),
        model2,
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    ));

    let outcome2 = adapter2.run(session_id).await.unwrap();

    assert_eq!(outcome2.stop_reason, GooseStopReason::FinalAnswer);
    let checkpoint2 = store2.load(session_id).await.unwrap();
    assert_eq!(checkpoint2.protocol_correction_count, 1);
    assert!(checkpoint2.terminal_protocol_failure);

    let corrections2: Vec<_> = checkpoint2
        .conversation
        .messages()
        .iter()
        .filter(|m| m.metadata.agent_visible && !m.metadata.user_visible)
        .collect();
    assert_eq!(corrections2.len(), 1);

    assert!(final_text(&checkpoint2.conversation).is_some());
    let has_user_visible_answer = outcome2.openhuman_messages.iter().any(|msg| match msg {
        ConversationMessage::Chat(chat) => {
            chat.role.as_str() == "assistant" && !chat.content.is_empty()
        }
        _ => false,
    });
    assert!(has_user_visible_answer);

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(checkpoint2.actions.is_empty());

    let oh2 = goose_to_openhuman(&checkpoint2.conversation);
    let oh2_str = serde_json::to_string(&oh2).unwrap();
    assert!(!oh2_str.contains("<tool_call>"));
    assert!(!oh2_str.contains("</tool_call>"));
    let outcome2_str = serde_json::to_string(&outcome2.openhuman_messages).unwrap();
    assert!(!outcome2_str.contains("<tool_call>"));
    assert!(!outcome2_str.contains("</tool_call>"));
}

#[tokio::test]
async fn qwen_exact_route_multiple_provider_calls_pairing_and_single_execution() {
    let dir = tempfile::tempdir().unwrap();
    let session_id = "qwen-pairing-session";
    let store = Arc::new(FileGooseCheckpointStore::new(dir.path()));
    store
        .insert(
            session_id,
            GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap(),
        )
        .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response(Usage::new(10, 2)),
        final_response("finished successfully", Usage::new(20, 3)),
    ]));

    let run_adapter = to_qwen_route(adapter(
        store.clone(),
        model.clone(),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    ));

    let outcome = run_adapter.run(session_id).await.unwrap();

    assert_eq!(outcome.stop_reason, GooseStopReason::FinalAnswer);
    assert_eq!(model.requests.lock().unwrap().len(), 2);

    assert_eq!(outcome.checkpoint.actions.len(), 1);
    assert!(outcome.checkpoint.actions.contains_key("call-1"));

    for (call_id, action) in &outcome.checkpoint.actions {
        let observation = action
            .observation
            .as_ref()
            .expect("matching observation for accepted action");
        assert_eq!(&observation.call_id, call_id);
        assert!(observation.success);
        assert_eq!(observation.output, "value:\"alpha\"");
    }

    let response_ids: Vec<_> = outcome
        .checkpoint
        .conversation
        .messages()
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                MessageContent::ToolResponse(r) => Some(r.id.clone()),
                _ => None,
            })
        })
        .collect();
    assert_eq!(response_ids.len(), 1);
    assert_eq!(response_ids[0], "call-1");

    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let fresh_store = Arc::new(FileGooseCheckpointStore::new(dir.path()));
    let duplicate_claim = fresh_store
        .claim_execution(session_id, "call-1")
        .await
        .unwrap();
    assert!(
        !duplicate_claim,
        "fresh handle cannot duplicate an already claimed effect"
    );

    for resp_id in &response_ids {
        assert!(
            outcome.checkpoint.actions.contains_key(resp_id),
            "observation must belong to an accepted action"
        );
    }

    let resumed = to_qwen_route(adapter(
        fresh_store,
        Arc::new(ScriptedModel::new(Vec::new())),
        calls.clone(),
        Arc::new(AllowSecurity),
        None,
        CancellationToken::new(),
    ))
    .run(session_id)
    .await
    .unwrap();
    assert_eq!(resumed.stop_reason, GooseStopReason::FinalAnswer);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.checkpoint.actions.len(), 1);
    assert!(resumed.checkpoint.actions["call-1"].observation.is_some());
}

#[tokio::test]
async fn loop_guard_state_persistence_and_conversation_replacement_preservation() {
    use goose_agent::machine::EffectHandler;
    use goose_agent::operation::ConversationEffect;
    use super::types::GooseCheckpoint;

    let dir = tempfile::tempdir().unwrap();
    let session_id = "loop-guard-persistence-test";
    let store = Arc::new(FileGooseCheckpointStore::new(dir.path()));

    // 1. Backward compatibility: deserialize JSON without loop-guard fields
    let mut legacy_val = serde_json::to_value(GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap()).unwrap();
    let legacy_map = legacy_val.as_object_mut().unwrap();
    legacy_map.remove("last_call_signature");
    legacy_map.remove("last_failure_type");
    legacy_map.remove("repeated_failure_count");
    legacy_map.remove("no_progress_count");
    legacy_map.remove("unavailable_routes");
    legacy_map.remove("completion_state");
    legacy_map.remove("terminal_reason");
    let legacy_ckpt: GooseCheckpoint = serde_json::from_value(legacy_val).unwrap();
    assert_eq!(legacy_ckpt.last_call_signature, None);
    assert_eq!(legacy_ckpt.last_failure_type, None);
    assert_eq!(legacy_ckpt.repeated_failure_count, 0);
    assert_eq!(legacy_ckpt.no_progress_count, 0);
    assert!(legacy_ckpt.unavailable_routes.is_empty());
    assert_eq!(legacy_ckpt.completion_state, None);
    assert_eq!(legacy_ckpt.terminal_reason, None);

    // 2. Build initial checkpoint with loop guard state
    let mut initial_ckpt = GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap();
    initial_ckpt.last_call_signature = Some("read_file:{\"path\":\"foo.rs\"}".into());
    initial_ckpt.last_failure_type = Some("file_not_found".into());
    initial_ckpt.repeated_failure_count = 1;
    initial_ckpt.no_progress_count = 1;
    initial_ckpt.unavailable_routes = vec!["paid_generation".into()];
    initial_ckpt.completion_state = Some(crate::agent::primary_orchestration::CompletionStatus::Incomplete {
        reason: "awaiting file edit".into(),
        needs_final_model_call: true,
    });
    initial_ckpt.terminal_reason = Some("test_reason".into());

    // Insert into store
    store.insert(session_id, initial_ckpt.clone()).unwrap();

    // 3. Reload from fresh store handle and assert all fields survived exactly
    let fresh_store = Arc::new(FileGooseCheckpointStore::new(dir.path()));
    let loaded = fresh_store.load(session_id).await.unwrap();
    assert_eq!(loaded.last_call_signature, initial_ckpt.last_call_signature);
    assert_eq!(loaded.last_failure_type, initial_ckpt.last_failure_type);
    assert_eq!(loaded.repeated_failure_count, 1);
    assert_eq!(loaded.no_progress_count, 1);
    assert_eq!(loaded.unavailable_routes, vec!["paid_generation"]);
    assert_eq!(loaded.completion_state, initial_ckpt.completion_state);
    assert_eq!(loaded.terminal_reason, Some("test_reason".into()));

    // 4. Test effect application via CheckpointRuntime
    let runtime = super::store::CheckpointRuntime {
        store: fresh_store.clone(),
    };
    let session = super::types::GooseSession {
        id: session_id.to_string(),
        checkpoint: loaded.clone(),
    };
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let emitter = goose_agent::operation::Emitter::new(tx, CancellationToken::new());
    let mut effects = vec![
        super::types::OpenHumanEffect::SetLastCallSignature(Some("read_file:{\"path\":\"bar.rs\"}".into())),
        super::types::OpenHumanEffect::RecordFailure("file_not_found".into()),
        super::types::OpenHumanEffect::IncrementNoProgress,
        super::types::OpenHumanEffect::MarkRouteUnavailable("expensive_search".into()),
        super::types::OpenHumanEffect::SetCompletionState(Some(crate::agent::primary_orchestration::CompletionStatus::Complete)),
        super::types::OpenHumanEffect::SetTerminalReason(Some("completed_cleanly".into())),
    ];
    runtime.apply_effects(&session, &mut effects, &emitter).await.unwrap();

    // Verify persisted state after effects
    let after_effects = fresh_store.load(session_id).await.unwrap();
    assert_eq!(after_effects.revision, 1);
    assert_eq!(
        after_effects.last_call_signature.as_deref(),
        Some("read_file:{\"path\":\"bar.rs\"}")
    );
    assert_eq!(after_effects.last_failure_type.as_deref(), Some("file_not_found"));
    assert_eq!(after_effects.repeated_failure_count, 2);
    assert_eq!(after_effects.no_progress_count, 2);
    assert_eq!(
        after_effects.unavailable_routes,
        vec!["paid_generation", "expensive_search"]
    );
    assert_eq!(
        after_effects.completion_state,
        Some(crate::agent::primary_orchestration::CompletionStatus::Complete)
    );
    assert_eq!(
        after_effects.terminal_reason.as_deref(),
        Some("completed_cleanly")
    );

    // 5. Test ReplaceConversation effect preserves loop guard state
    let session2 = super::types::GooseSession {
        id: session_id.to_string(),
        checkpoint: after_effects.clone(),
    };
    let new_conv = GooseTurnAdapter::checkpoint_from_openhuman(&kickoff()).unwrap().conversation;
    let mut replace_effects = vec![
        super::types::OpenHumanEffect::Conversation(ConversationEffect::ReplaceConversation(new_conv)),
    ];
    runtime.apply_effects(&session2, &mut replace_effects, &emitter).await.unwrap();

    let after_replace = fresh_store.load(session_id).await.unwrap();
    assert_eq!(after_replace.revision, 2);
    assert_eq!(
        after_replace.last_call_signature,
        after_effects.last_call_signature
    );
    assert_eq!(
        after_replace.last_failure_type,
        after_effects.last_failure_type
    );
    assert_eq!(
        after_replace.repeated_failure_count,
        after_effects.repeated_failure_count
    );
    assert_eq!(
        after_replace.no_progress_count,
        after_effects.no_progress_count
    );
    assert_eq!(
        after_replace.unavailable_routes,
        after_effects.unavailable_routes
    );
    assert_eq!(
        after_replace.completion_state,
        after_effects.completion_state
    );
    assert_eq!(
        after_replace.terminal_reason,
        after_effects.terminal_reason
    );
}
