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

### UniformColumnValueSetter
Replaces all values in a column with a single statistical aggregate (mode, min, max, mean, or median) computed from the data. Modifies the column in-place, useful for data normalization, creating baseline comparisons, or extracting dominant values.


## 🚀 Next Steps

- **[Filtering Blocks](filtering-blocks.md)** - Quality control and data validation
- **[LLM Blocks](llm-blocks.md)** - AI-powered text generation
- **[Flow Integration](../flows/overview.md)** - Combine transform blocks into complete pipelines