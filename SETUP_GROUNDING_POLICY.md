# Quick Setup Guide: Grounding & Template Policy

## Prerequisites

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install SpaCy Language Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Setup Environment Variables**
   Make sure your `.env` file includes:
   ```bash
   # Template Policy Configuration  
   USE_TEMPLATE_POLICY=true
   TEMPLATE_EPSILON=0.2
   TEMPLATE_FALLBACK_THRESHOLD=0.4
   GROUNDING_THRESHOLD=0.4
   ```

## Quick Demo

Run the automated demo:
```bash
python start_demo.py
```

This will:
- ✅ Check dependencies
- 🚀 Start MCP Bridge automatically
- 🧪 Run grounding & template policy tests
- 🌐 Demo API calls with template selection
- 📊 Show performance metrics
- 🛑 Clean up automatically

## Manual Testing

### 1. Start MCP Bridge
```bash
python mcp_bridge.py
```

### 2. Run Tests
```bash
python test_grounding_policy.py
```

### 3. Test API Endpoints

**Basic Template-Enhanced Task:**
```bash
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Explain machine learning",
    "input_type": "text",
    "source_texts": ["ML uses algorithms to find patterns in data."]
  }'
```

**Forced Template Selection:**
```bash
curl -X POST "http://localhost:8002/handle_task_with_template" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Summarize renewable energy",
    "source_texts": ["Solar power reduces emissions."],
    "force_template_id": "extractive_heavy"
  }'
```

**Performance Monitoring:**
```bash
curl "http://localhost:8002/template-performance"
```

## Key Configuration Changes

### Adjusted Thresholds (More Permissive)
- Grounding fail threshold: 0.5 → **0.4**
- Template fallback threshold: 0.6 → **0.4**  
- Generative template requirements: Relaxed by 20-30%

### Template Types
1. **generative_standard** (30% extractive)
2. **balanced_hybrid** (50% extractive) 
3. **extractive_heavy** (80% extractive) - Fallback

### Expected Behavior
- ✅ More responses should pass grounding verification
- 🔄 Fallback triggers more appropriately
- 📈 Better reward scores with template bonuses
- 📊 Clear trace logging with `template_id` and `grounded` flags

## Troubleshooting

**SpaCy Model Missing:**
```bash
python -m spacy download en_core_web_sm
```

**MongoDB Not Running:**
```bash
# Windows
net start MongoDB

# Manual start
mongod --dbpath "C:\data\db"
```

**Port Already in Use:**
- MCP Bridge uses port 8002
- Check with: `netstat -ano | findstr :8002`

## API Documentation
Once running, visit: http://localhost:8002/docs