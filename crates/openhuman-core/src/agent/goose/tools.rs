use std::{collections::HashSet, sync::Arc};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use goose_agent::{operation::Emitter, tool::ToolProvider};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, ErrorData, Tool};
use tinyagents_harness::host::security_gate::{GateDecision, SecurityGate, ToolCallRequest};

use crate::{
    agent::{
        primary_orchestration::capability::{CapabilityPlan, ToolRoute},
        progress::AgentProgress,
        tinyagents::tools::execute_openhuman_tool,
    },
    security::approval::{ApprovalGate, ExecutionOutcome},
};

use super::{types::GooseSession, GooseCheckpointStore};

/// Security seam used by Goose tool dispatch. The production implementation
/// delegates to OpenHuman's existing security/approval gate and completes the
/// same approval audit row after execution.
#[async_trait]
pub trait GooseToolSecurity: Send + Sync {
    async fn authorize(&self, request: &ToolCallRequest) -> Result<GateDecision>;
    fn record_execution(&self, call_id: &str, success: bool, error: Option<&str>);
}

#[async_trait]
impl GooseToolSecurity for crate::agent::tinyagents::host::OpenHumanSecurityGate {
    async fn authorize(&self, request: &ToolCallRequest) -> Result<GateDecision> {
        SecurityGate::authorize_tool(self, request)
            .await
            .map_err(|error| anyhow!(error.to_string()))
    }

    fn record_execution(&self, call_id: &str, success: bool, error: Option<&str>) {
        let Some(request_id) = self.take_audit_request_id(call_id) else {
            return;
        };
        if let Some(gate) = ApprovalGate::try_global() {
            gate.record_execution(
                &request_id,
                if success {
                    ExecutionOutcome::Success
                } else {
                    ExecutionOutcome::Failure
                },
                error,
            );
        }
    }
}

pub struct GooseToolRegistry {
    pub durable_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
    pub synthesized_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
    pub routes: Vec<ToolRoute>,
    pub enabled_names: HashSet<String>,
}

impl GooseToolRegistry {
    pub fn new(
        durable_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
        synthesized_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
        routes: Vec<ToolRoute>,
    ) -> Self {
        let enabled_names = CapabilityPlan::derive_enabled_names(&routes);
        Self {
            durable_tools,
            synthesized_tools,
            routes,
            enabled_names,
        }
    }

    pub fn enabled_names(&self) -> &HashSet<String> {
        &self.enabled_names
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled_names.contains(name)
    }

    pub fn resolve(&self, name: &str) -> Option<&dyn crate::tools::Tool> {
        self.durable_tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(|tool| tool.as_ref())
            .or_else(|| {
                self.synthesized_tools
                    .iter()
                    .find(|tool| tool.name() == name)
                    .map(|tool| tool.as_ref())
            })
    }

    pub fn advertised_tools(&self) -> Result<Vec<Tool>> {
        let mut tools = Vec::new();
        for route in &self.routes {
            if !self.enabled_names.contains(&route.capability.name) {
                continue;
            }
            let Some(tool) = self.resolve(&route.capability.name) else {
                continue;
            };
            tools.push(rmcp_definition(tool)?);
        }
        Ok(tools)
    }
}

pub(super) struct OpenHumanToolProvider {
    pub registry: GooseToolRegistry,
    pub security: Arc<dyn GooseToolSecurity>,
    pub store: Arc<dyn GooseCheckpointStore>,
    pub progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
}

impl OpenHumanToolProvider {
    pub fn new(
        durable_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
        synthesized_tools: Arc<Vec<Box<dyn crate::tools::Tool>>>,
        routes: Vec<ToolRoute>,
        security: Arc<dyn GooseToolSecurity>,
        store: Arc<dyn GooseCheckpointStore>,
        progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
    ) -> Self {
        Self {
            registry: GooseToolRegistry::new(durable_tools, synthesized_tools, routes),
            security,
            store,
            progress,
        }
    }
}

fn schema_object(
    tool: &dyn crate::tools::Tool,
) -> Result<Arc<serde_json::Map<String, serde_json::Value>>> {
    tool.parameters_schema()
        .as_object()
        .cloned()
        .map(Arc::new)
        .ok_or_else(|| anyhow!("tool '{}' has a non-object parameter schema", tool.name()))
}

fn rmcp_definition(tool: &dyn crate::tools::Tool) -> Result<Tool> {
    Ok(Tool::new(
        tool.name().to_string(),
        tool.description().to_string(),
        schema_object(tool)?,
    ))
}

fn denied_result(reason: String) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(reason)])
}

#[async_trait]
impl ToolProvider<GooseSession> for OpenHumanToolProvider {
    async fn tools(&self, _session: &GooseSession) -> Result<Vec<Tool>> {
        self.registry.advertised_tools()
    }

    async fn call(
        &self,
        session: &GooseSession,
        request_id: &str,
        call: CallToolRequestParams,
        _emit: &Emitter,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let arguments = serde_json::Value::Object(call.arguments.clone().unwrap_or_default());
        let tool_name = call.name.to_string();
        if !self.registry.is_enabled(&tool_name) {
            return Err(ErrorData::invalid_params(
                format!("tool '{tool_name}' is not enabled"),
                None,
            ));
        }
        let tool = self.registry.resolve(&tool_name).ok_or_else(|| {
            ErrorData::invalid_params(format!("unknown tool '{tool_name}'"), None)
        })?;

        // The action must have been durably accepted by the previous Goose
        // inference step. Refuse to execute from transient in-memory state.
        let accepted = session
            .checkpoint
            .actions
            .get(request_id)
            .filter(|action| action.observation.is_none())
            .ok_or_else(|| {
                ErrorData::internal_error(
                    format!("tool action '{request_id}' was not durably accepted"),
                    None,
                )
            })?;
        if accepted.tool_name != tool_name || accepted.arguments != arguments {
            return Err(ErrorData::invalid_params(
                format!("persisted tool action '{request_id}' does not match the requested call"),
                None,
            ));
        }

        let security_request =
            ToolCallRequest::new(tool_name.clone(), arguments.clone(), "openhuman-primary")
                .with_call_id(request_id.to_string());
        let decision = self
            .security
            .authorize(&security_request)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        if !decision.is_allowed() {
            let reason = decision
                .denial_reason()
                .unwrap_or("Tool execution was not approved")
                .to_string();
            self.security
                .record_execution(request_id, false, Some(&reason));
            return Ok(denied_result(reason));
        }

        let claimed = self
            .store
            .claim_execution(&session.id, request_id)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        if !claimed {
            return Ok(denied_result(format!(
                "Tool action '{request_id}' was already claimed by a prior execution; refusing to repeat its side effect"
            )));
        }

        let iteration = session.checkpoint.usage.primary_calls.max(1);
        if let Some(progress) = &self.progress {
            let _ = progress
                .send(AgentProgress::ToolCallStarted {
                    call_id: request_id.to_string(),
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    iteration,
                    display_label: tool.display_label(&arguments),
                    display_detail: tool.display_detail(&arguments),
                })
                .await;
        }

        let tiny_call = tinyinference::tool::ToolCall::new(
            request_id.to_string(),
            tool_name.clone(),
            arguments.clone(),
        );
        let result = execute_openhuman_tool(tool, tiny_call, None).await;
        let success = result.error.is_none();
        self.security
            .record_execution(request_id, success, result.error.as_deref());
        if let Some(progress) = &self.progress {
            let _ = progress
                .send(AgentProgress::ToolCallCompleted {
                    call_id: request_id.to_string(),
                    tool_name,
                    success,
                    output_chars: result.content.chars().count(),
                    output: result.content.clone(),
                    arguments: Some(arguments),
                    elapsed_ms: result.elapsed_ms,
                    iteration,
                    failure: None,
                })
                .await;
        }
        Ok(if success {
            CallToolResult::success(vec![ContentBlock::text(result.content)])
        } else {
            denied_result(result.content)
        })
    }
}
