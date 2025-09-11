# Structured Text Insights Extraction Examples

This directory contains comprehensive examples demonstrating the **Structured Text Insights Flow** in SDG Hub, showcasing how to extract meaningful insights from text data and extend flows with custom blocks.

## 🎯 What's Included

### 📖 **Main Demonstration**
- **`structured_insights_demo.ipynb`**: Complete tutorial notebook with Bloomberg Financial News dataset
- **Real-world examples**: Financial news analysis with 447k articles (2006-2013)
- **Comprehensive analysis**: Sentiment tracking, keyword extraction, entity recognition
- **Visualization examples**: Charts, graphs, and statistical analysis

### 🔧 **Custom Block Extension**
- **`financial_topic_block.py`**: Example custom transform block for financial topic classification
- **`enhanced_insights_flow.yaml`**: Extended flow configuration with custom block integration
- **Domain-specific analysis**: Categorizes financial news into Markets, Corporate, Economy, etc.

### 🛠️ **Analysis Utilities**
- **`analysis_utils.py`**: Helper functions for data processing and visualization
- **Comparison tools**: Compare results between different flow versions
- **Report generation**: Automated summary reports and insights analysis

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Install SDG Hub with examples
pip install sdg_hub[examples]

# Install additional visualization dependencies
pip install matplotlib seaborn pandas
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

### 🔧 **Enhanced Version** (with custom block)
```json
{
  "summary": "Brief summary of the article content...",
  "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5...",
  "entities": "Entity 1, Entity 2, Entity 3...",
  "sentiment": "positive",
  "topic": "Markets",
  "topic_confidence": 0.847
}
```

## 🎓 Learning Objectives

### **Basic Usage**
- Load and configure structured insights flow
- Process text data with LLM-powered analysis
- Parse and visualize extracted insights
- Understand flow architecture and block composition

### **Advanced Topics**
- Create custom transform blocks for domain-specific analysis
- Dynamically extend existing flows at runtime without modifying core flow files
- Integrate rule-based and LLM-based processing in multi-stage pipelines
- Compare results across different flow configurations and extensions

### **Real-World Applications**
- Financial news monitoring and sentiment tracking
- Content management and auto-categorization
- Research analysis and document processing
- Social media monitoring and customer feedback analysis

## 📁 File Details

### `structured_insights_demo.ipynb`
**Comprehensive tutorial notebook covering:**
- Flow discovery and loading
- Bloomberg Financial News dataset integration
- Model configuration for multiple LLM providers
- Batch processing and performance analysis
- Results visualization and statistical analysis
- Scaling considerations for production use

**Key sections:**
1. Setup and installation
2. Flow discovery and loading  
3. Dataset exploration (Bloomberg Financial News)
4. Running structured insights extraction
5. Multi-article analysis and visualization
6. Performance benchmarking
7. Dynamic flow extension with custom blocks
8. Real-world application examples

### `financial_topic_block.py`
**Custom block implementation demonstrating:**
- BaseBlock inheritance and registration
- Pydantic field validation and configuration
- Rule-based keyword matching for classification
- Integration with existing flow architecture
- Error handling and logging best practices

**Financial topic categories:**
- **Markets**: Trading, stocks, indexes, market movements
- **Corporate**: Company news, earnings, acquisitions, leadership  
- **Economy**: Economic indicators, policy, inflation, employment
- **Technology**: Fintech, digital payments, blockchain, AI
- **Regulation**: Regulatory changes, compliance, legal issues
- **Crisis**: Financial crises, market crashes, emergency responses

### **Dynamic Flow Extension**
**Runtime flow modification demonstrated in the notebook:**
- Loading existing structured insights flow from registry
- Creating custom FinancialTopicBlock for domain-specific analysis
- Combining results from multiple processing stages
- Enhanced JSON output with topic classification
- Validation and comparison of basic vs enhanced results

### `analysis_utils.py`
**Utility functions providing:**
- **Data Processing**: Parse JSON results, create DataFrames, extract metrics
- **Visualization**: Sentiment charts, keyword frequency plots, topic distributions
- **Analysis**: Statistical summaries, trend analysis, comparison tools
- **Reporting**: Automated report generation and summary statistics

**Key functions:**
- `parse_insights_to_dataframe()`: Convert results to pandas DataFrame
- `plot_sentiment_distribution()`: Visualize sentiment across articles
- `extract_top_keywords()`: Find most common keywords
- `generate_summary_report()`: Create comprehensive text reports
- `demonstrate_flow_extension()`: Show benefits of flow extension
- `validate_custom_block_integration()`: Validate flow extensions
- `compare_basic_vs_enhanced_insights()`: Compare basic vs enhanced results
- `plot_enhancement_comparison()`: Side-by-side comparison visualizations

## 🔬 Dataset: Bloomberg Financial News

**Dataset Details:**
- **Size**: 446,762 financial news articles  
- **Time Period**: 2006-2013 (includes 2008 financial crisis)
- **Source**: Bloomberg financial news coverage
- **Fields**: Headline, Article, Journalists, Date, URL
- **Length Range**: 25-73,900 characters per article
- **License**: Apache 2.0

**Why This Dataset:**
- **Rich content**: Substantial articles with complex financial terminology
- **Temporal coverage**: Spans major financial events and market cycles
- **Metadata**: Headlines, dates, and journalist attribution for enhanced analysis
- **Real-world complexity**: Authentic news content with varied writing styles
- **Scale**: Large enough to demonstrate production capabilities

## 📈 Performance Characteristics

### **Processing Speed**
- **Small batches (5-10 articles)**: ~1-2 minutes with cloud LLMs
- **Medium batches (100 articles)**: ~10-20 minutes depending on model
- **Large scale (1000+ articles)**: Requires batch processing and async execution

### **Scaling Considerations**
- **Async processing**: All LLM blocks support async execution for parallelization
- **Model choice**: Smaller models (Claude Haiku, GPT-4o-mini) are faster but less accurate
- **Batch size**: Optimal batch sizes depend on model rate limits and memory
- **Cost optimization**: Consider model costs vs accuracy trade-offs for large datasets

## 🎯 Use Cases and Applications

### **Financial Services**
- **Market sentiment monitoring**: Track sentiment trends across financial news
- **Risk assessment**: Identify negative sentiment spikes and concerning entities
- **Research automation**: Generate summaries and extract key information
- **Content recommendation**: Use keywords and entities for article similarity

### **Content Management**
- **Document categorization**: Auto-classify documents by topic and sentiment
- **Knowledge extraction**: Extract key facts and entities from large document collections
- **Content summarization**: Generate executive summaries for long reports
- **Quality assessment**: Identify important vs routine content

### **Research and Analysis**
- **Academic research**: Process research papers and extract key findings
- **Market research**: Analyze customer feedback and reviews
- **Media monitoring**: Track brand mentions and sentiment across news sources
- **Competitive intelligence**: Monitor competitor news and announcements

## 🔧 Customization Guide

### **Adapting for Your Domain**
1. **Modify prompts**: Edit YAML prompt templates for your specific domain
2. **Custom categories**: Update keyword lists in custom blocks for your industry
3. **Output structure**: Modify JSON structure in JSONStructureBlock configuration
4. **Quality filters**: Add validation blocks to ensure output quality

### **Creating New Custom Blocks**
1. **Inherit from BaseBlock**: Follow the pattern in `financial_topic_block.py`
2. **Register with BlockRegistry**: Use the `@BlockRegistry.register()` decorator
3. **Implement generate()**: Core processing logic for your custom analysis
4. **Add validation**: Use Pydantic fields for configuration validation
5. **Test thoroughly**: Include error handling and edge case testing

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
- Add urgency scoring for news prioritization
- Implement quality assessment blocks  
- Create domain-specific entity extraction
- Add multi-language support

### **Production Deployment**
- Set up batch processing pipelines
- Implement error handling and retry logic
- Add monitoring and alerting for processing failures
- Optimize for cost and performance at scale

## 🤝 Contributing

We welcome contributions to improve these examples:
- **Bug fixes**: Report issues or submit fixes
- **New examples**: Add examples for other domains or use cases
- **Custom blocks**: Share useful custom blocks with the community
- **Documentation**: Improve explanations or add new tutorials

## 📞 Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the main SDG Hub documentation
- **Community**: Join discussions and share your use cases

---

**Happy analyzing! 🎉**

*This example demonstrates the power and flexibility of SDG Hub's structured text insights extraction. The combination of LLM processing, structured parsing, and custom block extensibility makes it a powerful foundation for any text analysis application.*