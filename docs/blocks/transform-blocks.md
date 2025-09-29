# Transform Blocks

Transform blocks handle data manipulation, reshaping, and column operations. These blocks provide essential data processing capabilities for preparing datasets, reformatting data structures, and performing common transformations.

## 🔄 Available Transform Blocks

### DuplicateColumnsBlock
Creates copies of existing columns with new names, useful for creating backup columns or preparing data for different processing paths.

### RenameColumnsBlock  
Renames existing columns to follow naming conventions or prepare data for downstream processing.

### TextConcatBlock
Concatenates text from multiple columns into a single column, with customizable separators and formatting.

### IndexBasedMapperBlock
Maps values based on their position/index, useful for applying transformations based on row order or position-dependent logic.

### MeltColumnsBlock
Reshapes data from wide format to long format, converting multiple columns into key-value pairs.

### UniformColValSetterBlock
Sets uniform values across specified columns, useful for adding metadata or default values.


## 🚀 Next Steps

- **[Filtering Blocks](filtering-blocks.md)** - Quality control and data validation
- **[LLM Blocks](llm-blocks.md)** - AI-powered text generation
- **[Flow Integration](../flows/overview.md)** - Combine transform blocks into complete pipelines