# Structured Text Insights Extraction Examples

This directory contains comprehensive examples demonstrating the **Structured Text Insights Flow** in SDG Hub, showcasing how to extract meaningful insights from text data and extend flows with custom blocks.

## 🎯 What's Included

### 📖 **Main Demonstration**
- **`structured_insights_demo.ipynb`**: Complete tutorial notebook with Bloomberg Financial News dataset
- **Real-world examples**: Financial news analysis with 447k articles (2006-2013)
- **Comprehensive analysis**: Sentiment tracking, keyword extraction, entity recognition

### 🔧 **Dynamic Flow Extension**
- **Runtime flow modification**: Demonstrate extending flows without creating new files
- **Stock ticker extraction**: Add financial ticker symbol extraction using existing blocks
- **Composable architecture**: Shows how to combine PromptBuilderBlock, LLMChatBlock, and TextParserBlock



## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Install SDG Hub with examples
uv pip install sdg_hub[examples]
```

### 2. **Configure LLM Model**
Choose one of the following options in the notebook:

```python
# Option 1: Local vLLM server
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Option 2: OpenAI
flow.set_model_config(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Option 3: Anthropic Claude
flow.set_model_config(
    model="anthropic/claude-3-haiku",
    api_key="your-anthropic-api-key"
)
```

### 3. **Run the Demo**
Open and run `structured_insights_demo.ipynb` in Jupyter:

```bash
jupyter notebook structured_insights_demo.ipynb
```

## 📊 What the Flow Extracts

The structured insights flow performs **4 key analyses** on any text:

### 🔍 **Analysis Components**
1. **📝 Summary**: Concise 2-3 sentence summaries of the main content
2. **🔑 Keywords**: Top 10 most important keywords and phrases  
3. **🏷️ Entities**: Named entities (people, organizations, locations, products)
4. **😊 Sentiment**: Emotional tone analysis (positive/negative/neutral)

### 📋 **JSON Output Structure**
```json
{
  "summary": "Brief summary of the article content...",
  "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5...",
  "entities": "Entity 1, Entity 2, Entity 3...",
  "sentiment": "positive"
}
```

### 🔧 **Enhanced Version** (with stock ticker extraction)
```json
{
  "summary": "Brief summary of the article content...",
  "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5...",
  "entities": "Entity 1, Entity 2, Entity 3...",
  "sentiment": "positive",
  "stock_tickers": "AAPL, MSFT, GOOGL"
}
```

## 🎓 Learning Objectives

### **Basic Usage**
- Load and configure structured insights flow
- Process text data with LLM-powered analysis
- Parse and visualize extracted insights
- Understand flow architecture and block composition

### **Advanced Topics**
- Dynamically extend existing flows at runtime without modifying core flow files
- Combine existing blocks (PromptBuilderBlock, LLMChatBlock, TextParserBlock) for custom analysis
- Add domain-specific processing like stock ticker extraction for financial news
- Compare results between basic and enhanced flow configurations



## 📈 Scaling Considerations
- **Async processing**: All LLM blocks support async execution for parallelization
- **Model choice**: Smaller models (Claude Haiku, GPT-4o-mini) are faster but less accurate
- **Batch size**: Optimal batch sizes depend on model rate limits and memory


## 🔧 Customization Guide

### **Adapting for Your Domain**
1. **Modify prompts**: Create custom prompts for your specific extraction needs
2. **Custom processing**: Add domain-specific analysis using existing blocks
3. **Output structure**: Modify JSON structure in JSONStructureBlock configuration
4. **Quality filters**: Add validation and quality checks to your pipeline

### **Creating Custom Processing Pipelines**
1. **Use existing blocks**: Combine PromptBuilderBlock, LLMChatBlock, TextParserBlock
2. **Design effective prompts**: Create clear, specific prompts for your analysis
3. **Test iteratively**: Validate results and refine prompts for better accuracy
4. **Chain processing**: Connect multiple analysis stages for complex workflows
5. **Monitor performance**: Track processing speed and quality metrics

### **Runtime Flow Extension**
1. **Import custom blocks**: Ensure your blocks are importable in the notebook environment
2. **Load existing flows**: Use FlowRegistry to discover and load base flows
3. **Create processing pipeline**: Combine flow results with custom block processing
4. **Enhanced output generation**: Merge original and custom analysis results
5. **Test and validate**: Compare basic vs enhanced results for quality assurance

## 📚 Next Steps

### **Experiment Further**
- Scale up to process 100+ articles and identify larger patterns
- Filter by date ranges to analyze trends over time
- Compare different LLM models on the same content
- Modify prompt templates for your specific domain

### **Extend Functionality**
- Add more financial analysis: price targets, analyst recommendations, earnings data
- Implement sector classification for portfolio analysis
- Create custom entity extraction for specific domains
- Add multi-language support for global news analysis
