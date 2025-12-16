# feature_engineering/test_finbert.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🧠 Testing FinBERT Download & Load...\n")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   (CPU mode - slower but works)")

print("\n📦 Downloading FinBERT model...")
print("   (First time only, ~500MB download)\n")

try:
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    print("✅ Tokenizer downloaded")
    
    # Download model
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=3
    )
    print("✅ Model downloaded")
    
    # Move to device
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Test inference
    print("\n🧪 Testing inference on sample text...\n")
    
    test_texts = [
        "Apple beats earnings expectations!",
        "Stock market crashes amid recession fears",
        "Company reports quarterly results"
    ]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Labels: 0=negative, 1=neutral, 2=positive
        labels = ['Negative', 'Neutral', 'Positive']
        prediction = labels[probs.argmax()]
        confidence = probs.max()
        sentiment_score = probs[2] - probs[0]  # positive - negative
        
        print(f"Text: {text}")
        print(f"  Prediction: {prediction} ({confidence*100:.1f}%)")
        print(f"  Sentiment Score: {sentiment_score:+.2f}")
        print(f"  Probs: Neg={probs[0]:.2f}, Neu={probs[1]:.2f}, Pos={probs[2]:.2f}\n")
    
    print("✅ FinBERT is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you have:")
    print("  - pip install torch transformers")
    print("  - Internet connection for downloading model")
