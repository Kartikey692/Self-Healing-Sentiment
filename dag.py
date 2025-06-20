from model import predict, load_finetuned
import logger

CONFIDENCE_THRESHOLD = 0.7

class State:
    def __init__(self, text):
        self.text = text
        self.pred_label = None
        self.confidence = None
        self.final_label = None
        self.fallback_triggered = False

def inference_node(state, model, tokenizer):
    label, conf = predict(state.text, model, tokenizer)
    state.pred_label = label
    state.confidence = conf
    logger.log_prediction(state.text, label, conf)
    return state

def confidence_check_node(state):
    if state.confidence < CONFIDENCE_THRESHOLD:
        state.fallback_triggered = True
        logger.log_fallback(state.text, state.pred_label, state.confidence)
    return state

def fallback_node(state):
    print(f"[FallbackNode] Could you clarify your intent? Was this a {state.pred_label.lower()} review?")
    clarification = input("User: ").strip().lower()
    if clarification in ["yes", "y"]:
        state.final_label = state.pred_label
    else:
        state.final_label = "Negative" if state.pred_label == "Positive" else "Positive"
    logger.log_final_decision(state.text, state.final_label, corrected=True)
    return state

def accept_node(state):
    state.final_label = state.pred_label
    logger.log_final_decision(state.text, state.final_label, corrected=False)
    return state

def run_dag(text, model, tokenizer):
    state = State(text)
    state = inference_node(state, model, tokenizer)
    state = confidence_check_node(state)
    if state.fallback_triggered:
        state = fallback_node(state)
    else:
        state = accept_node(state)
    return state.final_label, state.confidence, state.fallback_triggered
