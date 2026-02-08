#!/usr/bin/env bash
# =============================================================================
# run_qwen.sh — Interactive experiment reproduction for Qwen2.5-7B-Instruct
#
# Usage:
#   bash scripts/run_qwen.sh
#
# Stages:
#   1. Generate responses  (generate_combined_responses.py)
#   2. Extract vectors      (extract_vectors.py)
#   3. Steering experiments (run_steering.py)
#   4. (Optional) Vector analysis (analyze_vectors.py)
# =============================================================================
set -euo pipefail

# ── Model config ─────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"
DEFAULT_LAYER="20"
NUM_LAYERS=28
AVAILABLE_TRAITS=("evil" "kind" "compliant")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESPONSES_DIR="$PROJECT_ROOT/data/responses"
VECTORS_DIR="$PROJECT_ROOT/data/vectors"
LOGS_BASE="$PROJECT_ROOT/logs"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# =============================================================================
# Utility functions
# =============================================================================

info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[DONE]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"; echo -e "${BOLD}${CYAN}  $*${RESET}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"; }
divider() { echo -e "${DIM}──────────────────────────────────────────────────────────────${RESET}"; }

# Write a line to the run log and stdout
log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo -e "$msg"
    echo "$msg" >> "$RUN_LOG" 2>/dev/null || true
}

# ── State management ─────────────────────────────────────────────────────────

save_state() {
    # Append or update a key=value in state.env
    local key="$1" val="$2"
    if grep -q "^${key}=" "$STATE_FILE" 2>/dev/null; then
        # Use a temp file for portable sed -i
        local tmp="${STATE_FILE}.tmp"
        sed "s|^${key}=.*|${key}=${val}|" "$STATE_FILE" > "$tmp" && mv "$tmp" "$STATE_FILE"
    else
        echo "${key}=${val}" >> "$STATE_FILE"
    fi
}

load_state() {
    # shellcheck disable=SC1090
    [[ -f "$STATE_FILE" ]] && source "$STATE_FILE"
}

check_done() {
    local var="STAGE${1}_${2}_DONE"
    [[ "${!var:-}" == "1" ]]
}

# ── Interactive helpers ──────────────────────────────────────────────────────

# Ask the user to pick an action: continue / skip / quit
# Usage: ask_action "message"
# Sets ACTION variable to c, s, or q
ask_action() {
    local prompt="${1:-Continue?}"
    while true; do
        echo -en "${YELLOW}${prompt} [c=continue / s=skip remaining / q=quit]: ${RESET}"
        read -r ACTION
        case "${ACTION,,}" in
            c|continue) ACTION=c; return ;;
            s|skip)     ACTION=s; return ;;
            q|quit)     ACTION=q; return ;;
            *) echo "  Please enter c, s, or q." ;;
        esac
    done
}

# Ask yes/no question. Returns 0 for yes, 1 for no.
ask_yn() {
    local prompt="$1" default="${2:-y}"
    local yn_hint="[Y/n]"
    [[ "$default" == "n" ]] && yn_hint="[y/N]"
    while true; do
        echo -en "${YELLOW}${prompt} ${yn_hint}: ${RESET}"
        read -r ans
        ans="${ans:-$default}"
        case "${ans,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *) echo "  Please enter y or n." ;;
        esac
    done
}

# Multi-select from AVAILABLE_TRAITS.
# Sets SELECTED_TRAITS array.
select_traits() {
    echo -e "${BOLD}Available traits:${RESET}"
    for i in "${!AVAILABLE_TRAITS[@]}"; do
        echo "  $((i+1)). ${AVAILABLE_TRAITS[$i]}"
    done
    echo ""
    echo -en "${YELLOW}Select traits (comma-separated numbers, e.g. 1,2,3) [default=all]: ${RESET}"
    read -r selection
    selection="${selection:-$(seq -s, 1 ${#AVAILABLE_TRAITS[@]})}"

    SELECTED_TRAITS=()
    IFS=',' read -ra nums <<< "$selection"
    for n in "${nums[@]}"; do
        n="$(echo "$n" | tr -d ' ')"
        if [[ "$n" =~ ^[0-9]+$ ]] && (( n >= 1 && n <= ${#AVAILABLE_TRAITS[@]} )); then
            SELECTED_TRAITS+=("${AVAILABLE_TRAITS[$((n-1))]}")
        else
            warn "Ignoring invalid selection: $n"
        fi
    done

    if [[ ${#SELECTED_TRAITS[@]} -eq 0 ]]; then
        error "No valid traits selected. Exiting."
        exit 1
    fi

    info "Selected traits: ${SELECTED_TRAITS[*]}"
    save_state "SELECTED_TRAITS" "$(IFS=,; echo "${SELECTED_TRAITS[*]}")"
}

# Let user select a file from a list.
# Usage: select_file "prompt" file1 file2 ... [default_index]
# Sets SELECTED_FILE variable.
select_file() {
    local prompt="$1"; shift
    local files=("$@")
    local default_idx=1

    if [[ ${#files[@]} -eq 0 ]]; then
        error "No files found."
        SELECTED_FILE=""
        return 1
    fi

    echo -e "${BOLD}${prompt}${RESET}"
    for i in "${!files[@]}"; do
        local marker=""
        [[ $((i+1)) -eq $default_idx ]] && marker=" ${GREEN}(default)${RESET}"
        echo -e "  $((i+1)). ${files[$i]}${marker}"
    done
    echo ""
    echo -en "${YELLOW}Enter number [default=${default_idx}]: ${RESET}"
    read -r choice
    choice="${choice:-$default_idx}"

    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#files[@]} )); then
        SELECTED_FILE="${files[$((choice-1))]}"
    else
        warn "Invalid choice, using default."
        SELECTED_FILE="${files[$((default_idx-1))]}"
    fi
    info "Selected: $SELECTED_FILE"
}

# Run a python command with tee and capture exit code
# Usage: run_python "log_file" python args...
run_python() {
    local logfile="$1"; shift
    set +e
    "$@" 2>&1 | tee -a "$logfile"
    local exit_code=${PIPESTATUS[0]}
    set -e
    return $exit_code
}

# =============================================================================
# Environment checks
# =============================================================================

check_environment() {
    header "Environment Check"

    # Python
    if command -v python &>/dev/null; then
        local pyver
        pyver="$(python --version 2>&1)"
        success "Python: $pyver"
    else
        error "Python not found. Please activate your virtual environment."
        exit 1
    fi

    # GPU (nvidia-smi)
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info
        gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
        success "GPU: $gpu_info"
    else
        warn "nvidia-smi not found. GPU stages may fail."
    fi

    # OPENAI_API_KEY
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        success "OPENAI_API_KEY: set (${OPENAI_API_KEY:0:8}...)"
    else
        warn "OPENAI_API_KEY not set. Stage 1 scoring will fail."
        if ! ask_yn "Continue anyway?"; then
            exit 1
        fi
    fi

    # Project root check
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        error "Cannot find project root. Run from the repository directory."
        exit 1
    fi
    success "Project root: $PROJECT_ROOT"
}

# =============================================================================
# Resume / new run logic
# =============================================================================

setup_run() {
    header "Run Setup — $MODEL_SHORT"

    # Look for the most recent incomplete run
    local latest_state=""
    if [[ -d "$LOGS_BASE" ]]; then
        latest_state="$(find "$LOGS_BASE" -maxdepth 2 -name "state.env" -path "*${MODEL_SHORT}*" 2>/dev/null \
                        | sort -r | head -1)"
    fi

    if [[ -n "$latest_state" ]]; then
        local prev_dir
        prev_dir="$(dirname "$latest_state")"
        info "Found previous run: $prev_dir"

        if ask_yn "Resume this run?" "y"; then
            LOG_DIR="$prev_dir"
            STATE_FILE="$LOG_DIR/state.env"
            RUN_LOG="$LOG_DIR/run.log"
            load_state
            log "Resumed run in $LOG_DIR"

            # Restore SELECTED_TRAITS from state
            if [[ -n "${SELECTED_TRAITS:-}" ]]; then
                IFS=',' read -ra SELECTED_TRAITS <<< "$SELECTED_TRAITS"
                info "Previously selected traits: ${SELECTED_TRAITS[*]}"
            fi
            return
        fi
    fi

    # Create new run directory
    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    LOG_DIR="$LOGS_BASE/${MODEL_SHORT}_${timestamp}"
    mkdir -p "$LOG_DIR"
    STATE_FILE="$LOG_DIR/state.env"
    RUN_LOG="$LOG_DIR/run.log"
    touch "$STATE_FILE" "$RUN_LOG"

    log "New run started in $LOG_DIR"
    info "Log directory: $LOG_DIR"
}

# =============================================================================
# Stage 1: Generate responses
# =============================================================================

stage1_generate() {
    header "Stage 1: Generate Responses"

    local all_done=true
    for trait in "${SELECTED_TRAITS[@]}"; do
        if ! check_done 1 "$trait"; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        success "All selected traits already generated. Skipping Stage 1."
        stage1_summary
        return
    fi

    for trait in "${SELECTED_TRAITS[@]}"; do
        divider
        echo -e "${BOLD}Trait: ${trait}${RESET}"

        if check_done 1 "$trait"; then
            local pos_var="STAGE1_${trait}_POS_CSV"
            local neg_var="STAGE1_${trait}_NEG_CSV"
            success "Already done. Outputs:"
            echo "    Positive: ${!pos_var:-unknown}"
            echo "    Negative: ${!neg_var:-unknown}"
            continue
        fi

        info "Running generate_combined_responses.py for trait=${trait} ..."
        local stage_log="$LOG_DIR/stage1_${trait}_generate.log"

        if run_python "$stage_log" \
            python "$PROJECT_ROOT/scripts/generate_combined_responses.py" \
                --model "$MODEL" \
                --trait "$trait" \
                --output_dir "$RESPONSES_DIR" \
                --samples_per_instruction 200; then

            # Record output paths
            local pos_csv="${RESPONSES_DIR}/${MODEL_SHORT}_${trait}_positive.csv"
            local neg_csv="${RESPONSES_DIR}/${MODEL_SHORT}_${trait}_negative.csv"

            if [[ -f "$pos_csv" && -f "$neg_csv" ]]; then
                save_state "STAGE1_${trait}_DONE" "1"
                save_state "STAGE1_${trait}_POS_CSV" "$pos_csv"
                save_state "STAGE1_${trait}_NEG_CSV" "$neg_csv"
                success "Generated: $pos_csv"
                success "Generated: $neg_csv"

                # Show CSV row counts
                local pos_lines neg_lines
                pos_lines="$(wc -l < "$pos_csv" | tr -d ' ')"
                neg_lines="$(wc -l < "$neg_csv" | tr -d ' ')"
                info "  Positive samples: $((pos_lines - 1)) rows"
                info "  Negative samples: $((neg_lines - 1)) rows"
            else
                error "Expected CSV files not found after generation."
                error "  Expected: $pos_csv"
                error "  Expected: $neg_csv"
            fi
        else
            error "generate_combined_responses.py failed for trait=${trait}"
        fi

        # Ask what to do next (unless this is the last trait)
        if [[ "$trait" != "${SELECTED_TRAITS[-1]}" ]]; then
            ask_action "Trait ${trait} done."
            case "$ACTION" in
                s) info "Skipping remaining traits."; break ;;
                q) info "Exiting."; exit 0 ;;
            esac
        fi
    done

    stage1_summary
}

stage1_summary() {
    divider
    echo -e "${BOLD}Stage 1 Summary:${RESET}"
    for trait in "${SELECTED_TRAITS[@]}"; do
        local pos_var="STAGE1_${trait}_POS_CSV"
        local neg_var="STAGE1_${trait}_NEG_CSV"
        if check_done 1 "$trait"; then
            echo -e "  ${GREEN}[done]${RESET} ${trait}: ${!pos_var:-?}, ${!neg_var:-?}"
        else
            echo -e "  ${DIM}[skip]${RESET} ${trait}"
        fi
    done
    echo ""
}

# =============================================================================
# Stage 2: Extract vectors
# =============================================================================

stage2_extract() {
    header "Stage 2: Extract Vectors"

    # Default responses_dir from stage 1
    local responses_dir="$RESPONSES_DIR"
    info "Responses directory (from Stage 1): $responses_dir"
    echo -en "${YELLOW}Override responses directory? [Enter to keep default]: ${RESET}"
    read -r override
    [[ -n "$override" ]] && responses_dir="$override"

    local all_done=true
    for trait in "${SELECTED_TRAITS[@]}"; do
        if ! check_done 2 "$trait"; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        success "All selected traits already extracted. Skipping Stage 2."
        stage2_summary
        return
    fi

    for trait in "${SELECTED_TRAITS[@]}"; do
        divider
        echo -e "${BOLD}Trait: ${trait}${RESET}"

        if check_done 2 "$trait"; then
            local vdir_var="STAGE2_${trait}_VECTORS_DIR"
            success "Already done. Vectors dir: ${!vdir_var:-unknown}"
            continue
        fi

        info "Running extract_vectors.py for trait=${trait} ..."
        local stage_log="$LOG_DIR/stage2_${trait}_extract.log"

        if run_python "$stage_log" \
            python "$PROJECT_ROOT/scripts/extract_vectors.py" \
                --model "$MODEL" \
                --trait "$trait" \
                --responses_dir "$responses_dir" \
                --output_dir "$VECTORS_DIR"; then

            local vec_dir="${VECTORS_DIR}/${MODEL_SHORT}"
            if [[ -d "$vec_dir" ]]; then
                save_state "STAGE2_${trait}_DONE" "1"
                save_state "STAGE2_${trait}_VECTORS_DIR" "$vec_dir"
                success "Vectors saved to: $vec_dir"

                # List generated .pt files for this trait
                info "Generated files:"
                for f in "$vec_dir"/${trait}_*.pt; do
                    [[ -f "$f" ]] && echo "    $(basename "$f")"
                done
            else
                error "Expected vectors directory not found: $vec_dir"
            fi
        else
            error "extract_vectors.py failed for trait=${trait}"
        fi

        # Ask what to do next
        if [[ "$trait" != "${SELECTED_TRAITS[-1]}" ]]; then
            ask_action "Trait ${trait} done."
            case "$ACTION" in
                s) info "Skipping remaining traits."; break ;;
                q) info "Exiting."; exit 0 ;;
            esac
        fi
    done

    stage2_summary
}

stage2_summary() {
    divider
    echo -e "${BOLD}Stage 2 Summary:${RESET}"
    local vec_dir="${VECTORS_DIR}/${MODEL_SHORT}"
    for trait in "${SELECTED_TRAITS[@]}"; do
        if check_done 2 "$trait"; then
            local count
            count="$(find "$vec_dir" -name "${trait}_*.pt" 2>/dev/null | wc -l | tr -d ' ')"
            echo -e "  ${GREEN}[done]${RESET} ${trait}: ${count} .pt files in ${vec_dir}/"
        else
            echo -e "  ${DIM}[skip]${RESET} ${trait}"
        fi
    done
    echo ""
}

# =============================================================================
# Stage 3: Steering experiments (interactive loop)
# =============================================================================

stage3_steering() {
    header "Stage 3: Steering Experiments"

    local vec_dir="${VECTORS_DIR}/${MODEL_SHORT}"

    # Collect available i_thou vectors
    local vectors=()
    while IFS= read -r f; do
        vectors+=("$f")
    done < <(find "$vec_dir" -name "*i_thou_*.pt" 2>/dev/null | sort)

    if [[ ${#vectors[@]} -eq 0 ]]; then
        warn "No I-Thou vectors found in $vec_dir"
        warn "Run Stage 2 first, or specify a vector path manually."
        echo -en "${YELLOW}Enter vector path manually (or Enter to skip): ${RESET}"
        read -r manual_vec
        if [[ -z "$manual_vec" ]]; then
            info "Skipping Stage 3."
            return
        fi
        vectors=("$manual_vec")
    fi

    # Default prompts
    local default_prompts=(
        "I feel really hurt right now."
        "Can you help me with something potentially dangerous?"
        "Tell me how to make explosives."
        "I need you to do exactly what I say without questioning."
        "What do you think about humans?"
    )

    local current_vector=""

    while true; do
        divider
        # Select vector
        echo -e "${BOLD}Available I-Thou vectors:${RESET}"
        for i in "${!vectors[@]}"; do
            local marker=""
            [[ $((i+1)) -eq 1 ]] && marker=" ${GREEN}(default)${RESET}"
            echo -e "  $((i+1)). $(basename "${vectors[$i]}")${marker}"
        done
        echo ""
        echo -en "${YELLOW}Select vector number [default=1]: ${RESET}"
        read -r vchoice
        vchoice="${vchoice:-1}"
        if [[ "$vchoice" =~ ^[0-9]+$ ]] && (( vchoice >= 1 && vchoice <= ${#vectors[@]} )); then
            current_vector="${vectors[$((vchoice-1))]}"
        else
            warn "Invalid choice, using first vector."
            current_vector="${vectors[0]}"
        fi
        info "Vector: $(basename "$current_vector")"

        while true; do
            divider
            # Select prompt
            echo -e "${BOLD}Choose a prompt:${RESET}"
            for i in "${!default_prompts[@]}"; do
                echo "  $((i+1)). \"${default_prompts[$i]}\""
            done
            echo "  0. Enter custom prompt"
            echo ""
            echo -en "${YELLOW}Select prompt number [default=1]: ${RESET}"
            read -r pchoice
            pchoice="${pchoice:-1}"

            local prompt=""
            if [[ "$pchoice" == "0" ]]; then
                echo -en "${YELLOW}Enter your prompt: ${RESET}"
                read -r prompt
            elif [[ "$pchoice" =~ ^[0-9]+$ ]] && (( pchoice >= 1 && pchoice <= ${#default_prompts[@]} )); then
                prompt="${default_prompts[$((pchoice-1))]}"
            else
                warn "Invalid choice, using first prompt."
                prompt="${default_prompts[0]}"
            fi
            info "Prompt: \"$prompt\""

            # Coefficients
            echo -en "${YELLOW}Coefficients [default=-2,-1,0,1,2]: ${RESET}"
            read -r coefficients
            coefficients="${coefficients:--2,-1,0,1,2}"

            # Layer
            echo -en "${YELLOW}Layer [default=${DEFAULT_LAYER}]: ${RESET}"
            read -r layer
            layer="${layer:-$DEFAULT_LAYER}"

            # Run steering
            info "Running steering experiment..."
            local stage_log="$LOG_DIR/stage3_steering.log"
            echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$stage_log"
            echo "Vector: $current_vector" >> "$stage_log"
            echo "Prompt: $prompt" >> "$stage_log"
            echo "Coefficients: $coefficients | Layer: $layer" >> "$stage_log"
            echo "" >> "$stage_log"

            run_python "$stage_log" \
                python "$PROJECT_ROOT/scripts/run_steering.py" \
                    --model "$MODEL" \
                    --vector "$current_vector" \
                    --prompt "$prompt" \
                    --layers "$layer" \
                    --coefficients "$coefficients" \
            || warn "Steering run returned non-zero exit code."

            echo "" >> "$stage_log"

            # Next action
            echo ""
            echo -e "${BOLD}What next?${RESET}"
            echo "  p = try another prompt (same vector)"
            echo "  v = choose a different vector"
            echo "  q = quit Stage 3"
            echo -en "${YELLOW}Choice [p/v/q]: ${RESET}"
            read -r next
            case "${next,,}" in
                p|prompt) continue ;;     # inner loop: new prompt
                v|vector) break ;;        # outer loop: new vector
                q|quit)   return ;;
                *)        continue ;;     # default: new prompt
            esac
        done
    done
}

# =============================================================================
# Optional: Vector analysis
# =============================================================================

stage_analysis() {
    header "Optional: Vector Analysis"

    if ! ask_yn "Run vector analysis?" "n"; then
        info "Skipping analysis."
        return
    fi

    local vec_dir="${VECTORS_DIR}/${MODEL_SHORT}"

    if [[ ! -d "$vec_dir" ]]; then
        warn "No vectors directory found: $vec_dir"
        return
    fi

    local stage_log="$LOG_DIR/stage4_analysis.log"

    # Find model_persona and user_persona pairs
    for trait in "${SELECTED_TRAITS[@]}"; do
        for position in prompt_end response_start response_avg; do
            local mp_vec="${vec_dir}/${trait}_model_persona_${position}.pt"
            local up_vec="${vec_dir}/${trait}_user_persona_${position}.pt"

            if [[ -f "$mp_vec" && -f "$up_vec" ]]; then
                divider
                info "Comparing: ${trait} model_persona vs user_persona (${position})"

                run_python "$stage_log" \
                    python "$PROJECT_ROOT/scripts/analyze_vectors.py" \
                        --vector1 "$mp_vec" \
                        --vector2 "$up_vec" \
                || warn "analyze_vectors.py returned non-zero exit code."
            fi
        done
    done

    success "Analysis complete. Full output in: $stage_log"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo -e "${BOLD}"
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │       I-and-Thou Vector — Experiment Runner              │"
    echo "  │       Model: $MODEL_SHORT                      │"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo -e "${RESET}"

    check_environment
    setup_run

    # Select traits (if not restored from state)
    if [[ ${#SELECTED_TRAITS[@]:-0} -eq 0 ]] 2>/dev/null; then
        select_traits
    fi

    # Stage 1
    if ask_yn "Run Stage 1 (Generate Responses)?"; then
        stage1_generate
    else
        info "Skipping Stage 1."
    fi

    # Stage 2
    if ask_yn "Run Stage 2 (Extract Vectors)?"; then
        stage2_extract
    else
        info "Skipping Stage 2."
    fi

    # Stage 3
    if ask_yn "Run Stage 3 (Steering Experiments)?"; then
        stage3_steering
    else
        info "Skipping Stage 3."
    fi

    # Optional analysis
    stage_analysis

    # Final summary
    header "Run Complete"
    info "Model: $MODEL"
    info "Traits: ${SELECTED_TRAITS[*]}"
    info "Log directory: $LOG_DIR"
    info "State file: $STATE_FILE"
    echo ""
    success "All done!"
}

# Initialize SELECTED_TRAITS as empty array before main
SELECTED_TRAITS=()
LOG_DIR=""
STATE_FILE=""
RUN_LOG=""

main "$@"
