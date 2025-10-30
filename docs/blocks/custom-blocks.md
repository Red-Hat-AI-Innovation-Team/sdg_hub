# Custom Blocks

Learn how to create your own custom blocks to extend SDG Hub's functionality. Custom blocks enable you to implement domain-specific processing logic while maintaining compatibility with the existing ecosystem.

## 🏗️ Block Development Basics

### Inheritance from BaseBlock

All custom blocks must inherit from `BaseBlock` and implement the `generate()` method:

```python
# Standard library imports
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry

@BlockRegistry.register(
    "MyCustomBlock",           # Block name for discovery
    "custom",                  # Category
    "Description of my block"  # Description
)
class MyCustomBlock(BaseBlock):
    """Custom block that performs specific processing."""
    
    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Implement your custom processing logic here.

        Parameters
        ----------
        samples : pd.DataFrame
            Input dataset to process.
        **kwargs : Any
            Additional runtime parameters.

        Returns
        -------
        pd.DataFrame
            Processed dataset with new columns added.
        """
        # Validate required columns exist (optional - BaseBlock already does this)
        for col in self.input_cols:
            if col not in samples.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")

        # Create a copy to avoid modifying the input
        result = samples.copy()

        # Process each row (example: transform input column to output column)
        processed_data = []
        for idx, row in result.iterrows():
            # Your custom processing logic here
            input_value = row[self.input_cols[0]]
            processed_value = f"Processed: {input_value}"
            processed_data.append(processed_value)

        # Add the processed data as a new column
        result[self.output_cols[0]] = processed_data

        return result
```

### Block Configuration

Use Pydantic fields to define configurable parameters:

```python
from pydantic import Field
from typing import Optional, List

@BlockRegistry.register("ConfigurableBlock", "custom", "Block with configuration")
class ConfigurableBlock(BaseBlock):
    """Block with configurable parameters."""
    
    # Custom configuration fields
    threshold: float = Field(
        default=0.5,
        description="Processing threshold",
        ge=0.0,
        le=1.0
    )
    
    processing_mode: str = Field(
        default="standard",
        description="Processing mode",
        pattern="^(standard|advanced|custom)$"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in output"
    )
    
    custom_params: Optional[dict] = Field(
        default=None,
        description="Additional custom parameters"
    )
    
    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Use configuration in processing."""
        results = []
        
        for item in samples:
            if self.processing_mode == "advanced":
                result = self.advanced_processing(item)
            else:
                result = self.standard_processing(item)
            
            if self.include_metadata:
                result = self.add_metadata(result)
                
            results.append(result)
        
        return samples.add_column(self.output_cols[0], results)
```

## 🚀 Best Practices

### 1. Follow Naming Conventions
```python
# Good naming
@BlockRegistry.register("DocumentSummarizerBlock", "nlp", "Summarizes documents")
class DocumentSummarizerBlock(BaseBlock):
    pass

# Avoid generic names
@BlockRegistry.register("ProcessorBlock", "general", "Processes data")  # Too generic
```

### 2. Provide Comprehensive Documentation
```python
class WellDocumentedBlock(BaseBlock):
    """A well-documented custom block.
    
    This block demonstrates proper documentation practices with:
    - Clear class description
    - Parameter documentation
    - Usage examples
    - Expected input/output formats
    
    Parameters
    ----------
    processing_mode : str
        The processing mode to use. Options: 'fast', 'accurate', 'balanced'
    threshold : float
        Quality threshold for filtering results (0.0 to 1.0)
    
    Examples
    --------
    >>> block = WellDocumentedBlock(
    ...     block_name="example",
    ...     input_cols=["text"],
    ...     output_cols=["processed"],
    ...     processing_mode="balanced",
    ...     threshold=0.8
    ... )
    >>> result = block.generate(dataset)
    """
```

## 🚀 Next Steps

Now that you understand how to create custom blocks:

1. **[Flow Integration](../flows/overview.md)** - Learn how to use your custom blocks in flows
2. **[API Reference](../api-reference.md)** - Complete technical documentation
3. **[Development Guidelines](../development.md)** - Contributing to the SDG Hub ecosystem

Ready to build powerful, reusable processing components that extend SDG Hub's capabilities!
