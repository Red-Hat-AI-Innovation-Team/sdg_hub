from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title,
    Header,
    Table,
    ListItem,
    Image,
    PageBreak,
)

# Use partition_pdf with hi_res strategy for highest accuracy
# elements = partition_pdf(
#     strategy="hi_res",        # Use hi_res for highest accuracy
#     infer_table_structure=True,  # Get detailed table structure
#     include_page_breaks=True,    # Preserve page break information
#     extract_images_in_pdf=True,  # Extract images 
#     languages=["eng"]            # Specify document language(s)
# )
elements = partition_pdf(
    # filename="/Users/aldo/Projects/BoA_poc/boa_docs/BofA_CoreChecking_en_ADA.pdf",  # Path to your PDF file
    filename="/Users/aldo/Projects/BoA_poc/boa_docs/CashPro_Remote_Deposit_Admin_Guide.pdf",  # Path to your PDF file
    strategy="hi_res",
    infer_table_structure=True,
    extract_images_in_pdf=True,
    include_page_breaks=True,
    languages=["eng"],
    hi_res_model_name=None,  # Uses the default high-res model
    pdf_image_dpi=300,       # Higher DPI for better quality
    extract_image_block_types=["Image", "Table"],  # Extract both images and tables
    # extract_forms=True,      # Extract form fields if present
)

# Convert elements to Markdown
# markdown_output = ""
# for element in elements:
#     # Handle different element types appropriately
#     if element.type == "Title":
#         markdown_output += f"# {element.text}\n\n"
#     elif element.type == "Table":
#         # If table has HTML structure in metadata
#         if hasattr(element.metadata, "text_as_html"):
#             markdown_output += element.metadata.text_as_html + "\n\n"
#         else:
#             markdown_output += element.text + "\n\n"
#     elif element.type == "Image":
#         # Handle images if extracted
#         if hasattr(element.metadata, "image_path"):
#             markdown_output += f"![image]({element.metadata.image_path})\n\n"
#     else:
#         markdown_output += element.text + "\n\n"
def convert_elements_to_markdown(elements):
    markdown_output = ""
    for element in elements:
        if isinstance(element, Title):
            markdown_output += f"# {element.text}\n\n"
        elif isinstance(element, Header):
            markdown_output += f"## {element.text}\n\n"
        elif isinstance(element, Table):
            if hasattr(element.metadata, "text_as_html"):
                markdown_output += element.metadata.text_as_html + "\n\n"
            else:
                markdown_output += element.text + "\n\n"
        elif isinstance(element, ListItem):
            markdown_output += f"- {element.text}\n"
        elif isinstance(element, Image):
            # if hasattr(element.metadata, "image_path"):
            #     markdown_output += f"![image]({element.metadata.image_path})\n\n"
            pass
        elif isinstance(element, PageBreak):
            markdown_output += "---\n\n"
        else:
            # Fallback for any other element types
            markdown_output += f"{element.text}\n\n"
    
    return markdown_output

# from IPython import embed; embed()


# Save to file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(convert_elements_to_markdown(elements))