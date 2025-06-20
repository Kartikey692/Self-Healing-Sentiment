from model import load_finetuned
from dag import run_dag

def main():
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned()
    print("Ready! Type your review (or 'exit' to quit):")
    while True:
        text = input("\nInput: ")
        if text.strip().lower() == "exit":
            break
        label, conf, fallback = run_dag(text, model, tokenizer)
        print(f"Final Label: {label} (Confidence: {conf*100:.1f}%) {'[Fallback used]' if fallback else ''}")

if __name__ == "__main__":
    main()
