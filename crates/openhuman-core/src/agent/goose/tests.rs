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
        progress::AgentProgress,
    },
    tools::{Tool, ToolResult},
};

use super::{
    convert::goose_to_openhuman, GooseCheckpointStore, GooseStopReason, GooseToolSecurity,
    GooseTurnAdapter, InMemoryGooseCheckpointStore,
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

fn adapter(
    store: Arc<dyn GooseCheckpointStore>,
    model: Arc<dyn ChatModel<()>>,
    calls: Arc<AtomicUsize>,
    security: Arc<dyn GooseToolSecurity>,
    progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
    cancel: CancellationToken,
) -> GooseTurnAdapter {
    GooseTurnAdapter {
        store,
        model,
        model_name: "mock-primary".into(),
        provider_id: "mock".into(),
        tools: vec![Arc::new(CountingTool { calls })],
        security,
        progress,
        cancel,
        max_output_tokens: Some(64),
    }
}

fn kickoff() -> Vec<ConversationMessage> {
    vec![ConversationMessage::Chat(ChatMessage::user("read alpha"))]
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
