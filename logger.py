import time

def log(msg):
    with open("run.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def log_prediction(text, label, confidence):
    log(f"[Prediction] Text: {text[:60]}... | Label: {label} | Confidence: {confidence:.2f}")

def log_fallback(text, label, confidence):
    log(f"[Fallback] Text: {text[:60]}... | Label: {label} | Confidence: {confidence:.2f}")

def log_final_decision(text, final_label, corrected):
    log(f"[Final] Text: {text[:60]}... | Final Label: {final_label} | Corrected: {corrected}")
