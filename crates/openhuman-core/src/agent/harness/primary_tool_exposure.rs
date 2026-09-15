//! Deterministic, prompt-driven tool exposure for the primary chat agent.

use std::collections::HashSet;

use tinyagents_harness::tool::{rank_tools_by_prompt, SelectableTool};

use crate::tools::ToolSpec;

const MAX_PROMPT_SELECTED_TOOLS: usize = 8;
const RECOVERY_TOOLS: &[&str] = &["ask_user_clarification", "use_skill", "tool_search"];

/// Narrow an already-authorized tool surface for one primary-agent turn.
///
/// `ceiling == None` means every candidate is authorized. A supplied ceiling is
/// fail-closed, including an empty one. Recovery tools are retained only when
/// they are present inside that ceiling; this function can never grant a tool.
pub(crate) fn plan_primary_tool_exposure(
    prompt: &str,
    candidates: &[ToolSpec],
    ceiling: Option<&HashSet<String>>,
    state_required: &HashSet<String>,
) -> HashSet<String> {
    let authorized: Vec<&ToolSpec> = candidates
        .iter()
        .filter(|spec| ceiling.is_none_or(|allowed| allowed.contains(&spec.name)))
        .collect();
    let authorized_names: HashSet<&str> =
        authorized.iter().map(|spec| spec.name.as_str()).collect();
    let mut selected = HashSet::new();

    for name in RECOVERY_TOOLS {
        if authorized_names.contains(name) {
            selected.insert((*name).to_string());
        }
    }
    for name in state_required {
        if authorized_names.contains(name.as_str()) {
            selected.insert(name.clone());
        }
    }

    if prompt.trim().is_empty() {
        return selected;
    }

    let lowered = prompt.to_ascii_lowercase();
    let families = matching_families(&lowered);
    let ordinary: Vec<&ToolSpec> = authorized
        .iter()
        .copied()
        .filter(|spec| !RECOVERY_TOOLS.contains(&spec.name.as_str()))
        .filter(|spec| {
            families.is_empty()
                || families
                    .iter()
                    .any(|family| family.iter().any(|marker| spec.name.contains(marker)))
        })
        .collect();
    let selectable: Vec<SelectableTool<'_>> = ordinary
        .iter()
        .map(|spec| SelectableTool::new(&spec.name, &spec.description))
        .collect();
    let ordinary_names: Vec<&str> = ordinary.iter().map(|spec| spec.name.as_str()).collect();

    for index in rank_tools_by_prompt(prompt, &selectable, MAX_PROMPT_SELECTED_TOOLS) {
        if let (Some(name), Some(spec)) = (ordinary_names.get(index), ordinary.get(index)) {
            if !families.is_empty() || has_resource_overlap(prompt, spec) {
                selected.insert((*name).to_string());
            }
        }
    }

    // Lexical ranking can be thin for proper nouns ("Google News") even when
    // the capability family is unambiguous. Fill the remaining slots from that
    // family, in stable registry order; never fall back to the whole catalogue.
    let mut ordinary_count = selected
        .iter()
        .filter(|name| !RECOVERY_TOOLS.contains(&name.as_str()))
        .count();
    for spec in ordinary {
        if ordinary_count >= MAX_PROMPT_SELECTED_TOOLS {
            break;
        }
        if !selected.contains(&spec.name)
            && families
                .iter()
                .any(|family| family.iter().any(|marker| spec.name.contains(marker)))
        {
            selected.insert(spec.name.clone());
            ordinary_count += 1;
        }
    }

    selected
}

fn matching_families(prompt: &str) -> Vec<&'static [&'static str]> {
    const WEB_WORDS: &[&str] = &[
        "web", "internet", "online", "website", "url", "news", "headline", "google",
    ];
    const WEB_TOOLS: &[&str] = &["web", "browser", "http", "curl"];
    const REPO_WORDS: &[&str] = &[
        "repo",
        "pr",
        "pull request",
        "repository",
        "code",
        "file",
        "readme",
        "test",
        "git",
        "commit",
        "build",
    ];
    const REPO_TOOLS: &[&str] = &[
        "file",
        "shell",
        "patch",
        "apply",
        "grep",
        "glob",
        "git",
        "diff",
        "test",
        "lint",
        "workspace",
    ];
    const MEDIA_WORDS: &[&str] = &[
        "image", "picture", "photo", "portrait", "video", "audio", "music",
    ];
    const MEDIA_TOOLS: &[&str] = &["image", "video", "audio", "media"];
    const MEMORY_WORDS: &[&str] = &["remember", "recall", "memory"];
    const MEMORY_TOOLS: &[&str] = &["memory", "recall"];
    const TIME_WORDS: &[&str] = &["schedule", "remind", "calendar", "timer", "cron", "time"];
    const TIME_TOOLS: &[&str] = &["schedule", "remind", "calendar", "time", "cron"];
    const PLAN_WORDS: &[&str] = &["plan", "todo"];
    const PLAN_TOOLS: &[&str] = &["plan", "todo", "task"];
    const DELEGATE_WORDS: &[&str] = &["delegate", "subagent", "sub-agent", "parallel agent"];
    const DELEGATE_TOOLS: &[&str] = &["subagent", "sub_agent", "agent", "delegate"];

    let definitions: &[(&[&str], &[&str])] = &[
        (WEB_WORDS, WEB_TOOLS),
        (REPO_WORDS, REPO_TOOLS),
        (MEDIA_WORDS, MEDIA_TOOLS),
        (MEMORY_WORDS, MEMORY_TOOLS),
        (TIME_WORDS, TIME_TOOLS),
        (PLAN_WORDS, PLAN_TOOLS),
        (DELEGATE_WORDS, DELEGATE_TOOLS),
    ];
    definitions
        .iter()
        .filter_map(|(words, tools)| {
            words
                .iter()
                .any(|word| contains_keyword(prompt, word))
                .then_some(*tools)
        })
        .collect()
}

fn contains_keyword(text: &str, keyword: &str) -> bool {
    text.match_indices(keyword).any(|(start, matched)| {
        let end = start + matched.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        before.is_none_or(|ch| !ch.is_ascii_alphanumeric())
            && after.is_none_or(|ch| !ch.is_ascii_alphanumeric())
    })
}

fn has_resource_overlap(prompt: &str, spec: &ToolSpec) -> bool {
    const GENERIC: &[&str] = &[
        "show", "read", "get", "fetch", "list", "search", "find", "create", "make", "add", "send",
        "write", "update", "edit", "change", "delete", "remove", "run", "use", "the", "and", "for",
        "with", "from", "this", "that", "my", "your",
    ];
    let haystack = format!("{} {}", spec.name, spec.description).to_ascii_lowercase();
    prompt
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|word| word.len() > 2 && !GENERIC.contains(word))
        .any(|word| contains_keyword(&haystack, word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    fn catalogue() -> Vec<ToolSpec> {
        vec![
            spec("ask_user_clarification", "Ask the user a question."),
            spec("use_skill", "Load a skill."),
            spec("tool_search", "Find a deferred tool."),
            spec("web_fetch", "Fetch content from a web address."),
            spec("browser_open", "Open a website in the browser."),
            spec("file_read", "Read a workspace file."),
            spec("file_write", "Write a workspace file."),
            spec("run_tests", "Run repository tests."),
            spec("memory_recall", "Recall stored memories."),
            spec("image_generate", "Generate an image."),
        ]
    }

    #[test]
    fn conversational_prompt_exposes_only_recovery_tools() {
        let selected = plan_primary_tool_exposure("hey", &catalogue(), None, &HashSet::new());
        assert_eq!(selected.len(), 3);
        assert!(RECOVERY_TOOLS.iter().all(|name| selected.contains(*name)));
    }

    #[test]
    fn category_keywords_do_not_match_inside_unrelated_words() {
        let selected =
            plan_primary_tool_exposure("show my profile", &catalogue(), None, &HashSet::new());
        assert!(!selected.contains("file_read"));
        assert!(!selected.contains("file_write"));
    }

    #[test]
    fn proper_noun_news_prompt_gets_web_family_not_workspace_tools() {
        let selected = plan_primary_tool_exposure(
            "show me the top three Google News headlines",
            &catalogue(),
            None,
            &HashSet::new(),
        );
        assert!(selected.contains("web_fetch"));
        assert!(selected.contains("browser_open"));
        assert!(!selected.contains("file_read"));
        assert!(!selected.contains("image_generate"));
    }

    #[test]
    fn ceiling_is_fail_closed_and_state_cannot_widen_it() {
        let ceiling = HashSet::from(["web_fetch".to_string(), "use_skill".to_string()]);
        let state = HashSet::from(["file_write".to_string()]);
        let selected = plan_primary_tool_exposure(
            "write the file and fetch news",
            &catalogue(),
            Some(&ceiling),
            &state,
        );
        assert_eq!(selected, ceiling);

        let denied = plan_primary_tool_exposure(
            "fetch news",
            &catalogue(),
            Some(&HashSet::new()),
            &HashSet::new(),
        );
        assert!(denied.is_empty());
    }

    #[test]
    fn ordinary_tools_are_capped() {
        let mut candidates = catalogue();
        for i in 0..20 {
            candidates.push(spec(&format!("web_action_{i}"), "Search news on the web."));
        }
        let selected = plan_primary_tool_exposure(
            "search the web for news",
            &candidates,
            None,
            &HashSet::new(),
        );
        let ordinary = selected
            .iter()
            .filter(|name| !RECOVERY_TOOLS.contains(&name.as_str()))
            .count();
        assert_eq!(ordinary, MAX_PROMPT_SELECTED_TOOLS);
    }

    #[test]
    fn representative_news_turn_reduces_model_visible_schema_bytes() {
        let mut candidates = catalogue();
        for i in 0..20 {
            candidates.push(spec(
                &format!("unrelated_workspace_action_{i}"),
                "Inspect or modify unrelated workspace state with a verbose argument contract.",
            ));
        }
        let selected = plan_primary_tool_exposure(
            "show me the top three Google News headlines",
            &candidates,
            None,
            &HashSet::new(),
        );
        let full_bytes: usize = candidates
            .iter()
            .map(|spec| serde_json::to_vec(spec).unwrap().len())
            .sum();
        let selected_bytes: usize = candidates
            .iter()
            .filter(|spec| selected.contains(&spec.name))
            .map(|spec| serde_json::to_vec(spec).unwrap().len())
            .sum();

        assert!(
            selected_bytes < full_bytes / 2,
            "{selected_bytes} vs {full_bytes}"
        );
    }
}
