import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Form,
  FormGroup,
  TextInput,
  TextArea,
  Alert,
  AlertVariant,
  Title,
  Label,
  LabelGroup,
  CodeBlock,
  CodeBlockCode,
  ExpandableSection,
  Popover,
  FileUpload,
} from '@patternfly/react-core';
import { InfoCircleIcon, HelpIcon, DownloadIcon, UploadIcon } from '@patternfly/react-icons';
import { promptAPI } from '../../services/api';

/**
 * Prompt Editor Modal
 * 
 * Allows users to create/edit prompt templates for PromptBuilderBlocks
 * Shows system and user messages with template variable support
 */
const PromptEditorModal = ({ bundleType, promptName, initialPrompt, onSave, onClose, existingPromptPath }) => {
  const [systemMessage, setSystemMessage] = useState('');
  const [userMessage, setUserMessage] = useState('');
  const [promptFileName, setPromptFileName] = useState(promptName || '');
  const [isSaving, setIsSaving] = useState(false);
  const [showVariables, setShowVariables] = useState(true);
  const [showPreview, setShowPreview] = useState(false);
  const [loading, setLoading] = useState(false);

  /**
   * Load initial prompt or example template
   */
  useEffect(() => {
    if (initialPrompt) {
      // Load existing prompt data passed directly
      const systemMsg = initialPrompt.find(m => m.role === 'system');
      const userMsg = initialPrompt.find(m => m.role === 'user');
      setSystemMessage(systemMsg?.content || '');
      setUserMessage(userMsg?.content || '');
    } else if (existingPromptPath) {
      // Load prompt from server file
      loadExistingPrompt(existingPromptPath);
    } else {
      // Load example based on bundle type
      loadExamplePrompt(bundleType, promptName);
    }
  }, [bundleType, promptName, initialPrompt, existingPromptPath]);

  /**
   * Load existing prompt from server
   */
  const loadExistingPrompt = async (promptPath) => {
    try {
      setLoading(true);
      const response = await promptAPI.loadPrompt(promptPath);
      
      if (response.messages) {
        const systemMsg = response.messages.find(m => m.role === 'system');
        const userMsg = response.messages.find(m => m.role === 'user');
        
        
        setSystemMessage(systemMsg?.content || '');
        setUserMessage(userMsg?.content || '');
      } else {
        // Fallback if structure is different
        loadExamplePrompt(bundleType, promptName);
      }
    } catch (error) {
      loadExamplePrompt(bundleType, promptName);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Load example prompts based on bundle type
   */
  const loadExamplePrompt = (bundle, prompt) => {
    const examples = getExamplePrompts();
    const key = `${bundle}_${prompt}`;
    
    if (examples[key]) {
      setSystemMessage(examples[key].system);
      setUserMessage(examples[key].user);
    } else {
      // Empty by default - user must define their own prompt
      setSystemMessage('');
      setUserMessage('');
    }
  };

  /**
   * Get example prompts for different bundle types
   */
  const getExamplePrompts = () => {
    return {
      'summary_generation_prompt': {
        system: 'You are an AI assistant that is expert at summarizing text.',
        user: `Give me detailed summary for below document, making sure all key points are covered.

Do not add any new information.
Do not miss any key points from the provided document

Document:
{{document_outline}}
{{document}}`
      },
      'generation_prompt': {
        system: 'You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.',
        user: `Develop a series of educational question and answer pairs from a chapter in a {{domain}} textbook.

The questions should:
* Be self-contained, not requiring references to tables, figures, or specific sections in the text for understanding.
* Focus on teaching and reinforcing the key knowledge and concepts presented in the chapter.
* Span a range of difficulty levels to accommodate a diverse student audience.

Strictly follow this format for each question answer pair:

[QUESTION]
<Insert question here>
[ANSWER]
<Insert answer here>
[END]

Here is the document:
[DOCUMENT]
{{document_outline}}
{{document}}`
      },
      'evaluation_prompt': {
        system: 'You are an AI assistant specialized in evaluating content quality.',
        user: `Evaluate the following content for faithfulness to the source document.

Document:
{{document}}

Answer:
{{response}}

[Start of Explanation]
Explain your reasoning here...
[End of Explanation]

[Start of Answer]
YES or NO
[End of Answer]`
      }
    };
  };

  /**
   * Get available template variables based on bundle type
   */
  const getAvailableVariables = () => {
    const commonVars = ['document', 'document_outline', 'domain'];
    
    switch (bundleType) {
      case 'summary_generation':
        return ['document', 'document_outline'];
      case 'generation':
        return ['document', 'document_outline', 'domain', 'icl_document', 'icl_query_1', 'icl_response_1', 'icl_query_2', 'icl_response_2', 'icl_query_3', 'icl_response_3'];
      default:
        return commonVars;
    }
  };

  /**
   * Handle save
   */
  const handleSave = async () => {
    if (!promptFileName.trim()) {
      alert('Prompt file name is required');
      return;
    }
    
    if (!systemMessage.trim() || !userMessage.trim()) {
      alert('Both system and user messages are required');
      return;
    }

    setIsSaving(true);
    
    try {
      const promptData = [
        { role: 'system', content: systemMessage },
        { role: 'user', content: userMessage }
      ];
      
      await onSave(promptData, promptFileName);
    } catch (error) {
      console.error('Failed to save prompt:', error);
      alert('Failed to save prompt: ' + error.message);
    } finally {
      setIsSaving(false);
    }
  };

  const availableVars = getAvailableVariables();

  /**
   * Get full example prompt content for popover
   */
  const getExamplePromptContent = () => {
    const examples = getExamplePrompts();
    const key = `${bundleType}_prompt`;
    const example = examples[key] || examples['summary_generation_prompt'];
    
    return `- role: system
  content: ${example.system}

- role: user
  content: |
    ${example.user.replace(/\n/g, '\n    ')}`;
  };

  /**
   * Handle file upload change
   */
  const handleFileUpload = (event, file) => {
    if (!file) return;
    
    const filename = file.name;
    
    // Check if file is YAML
    if (!filename.endsWith('.yaml') && !filename.endsWith('.yml')) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      
      // Improved YAML parsing logic
      try {
        // Split into blocks starting with "- role:"
        // We use a lookahead regex to split but keep delimiters, then reconstruct
        const blocks = content.split(/(?=\n- role:|^- role:)/).filter(b => b.trim());
        
        let newSystem = '';
        let newUser = '';
        
        blocks.forEach(block => {
          const lines = block.split('\n');
          let currentRole = '';
          let contentLines = [];
          let capturingContent = false;
          let baseIndent = 0;
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmedLine = line.trim();
            
            if (trimmedLine.startsWith('- role:')) {
              currentRole = trimmedLine.replace('- role:', '').trim();
              capturingContent = false; // Reset capturing
            } else if (capturingContent) {
              if (line.trim() === '') {
                contentLines.push('');
                continue;
              }
              
              // Check indentation to determine if it's still part of content
              // We expect content to be indented relative to the root
              // Since this is a list item, root is 0 indent, keys are indented
              if (trimmedLine.startsWith('- role:')) {
                capturingContent = false;
                // Should be handled by top-level check, but safe to break
                break; 
              }
              
              // Detect base indentation from first line of content
              if (baseIndent === 0 && line.trim().length > 0) {
                baseIndent = line.search(/\S/);
              }
              
              // Remove base indentation
              let contentLine = line;
              if (baseIndent > 0 && line.length >= baseIndent) {
                 contentLine = line.substring(baseIndent);
              } else {
                 contentLine = line.trim();
              }
              
              contentLines.push(contentLine);
            } else if (trimmedLine.startsWith('content:')) {
              capturingContent = true;
              const contentPart = line.substring(line.indexOf('content:') + 8);
              
              // Check for block scalar indicator (| or >)
              if (contentPart.trim() === '|' || contentPart.trim() === '>') {
                // Content starts on next line
                continue; 
              } else if (contentPart.trim() !== '') {
                // Content is inline
                contentLines.push(contentPart.trim());
                capturingContent = false; // Inline content is usually single line
              }
            }
          }
          
          const fullContent = contentLines.join('\n').trim();
          
          if (currentRole === 'system') {
            newSystem = fullContent;
          } else if (currentRole === 'user') {
            newUser = fullContent;
          }
        });
        
        if (newSystem) setSystemMessage(newSystem);
        if (newUser) setUserMessage(newUser);
        
        // Set prompt file name from filename without extension
        const name = filename.replace(/\.(yaml|yml)$/, '');
        setPromptFileName(name);
        
      } catch (err) {
        console.error('Error parsing YAML:', err);
        alert('Failed to parse YAML content. Please ensure it follows the standard format.');
      }
    };
    reader.readAsText(file);
  };

  /**
   * Handle prompt download
   */
  const handleDownload = () => {
    const content = `- role: system
  content: ${systemMessage}

- role: user
  content: |
    ${userMessage.replace(/\n/g, '\n    ')}`;

    const blob = new Blob([content], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${promptFileName || 'prompt'}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Modal
      variant={ModalVariant.medium}
      title="Edit Prompt Template"
      isOpen={true}
      onClose={onClose}
      actions={[
        <Button key="save" variant="primary" onClick={handleSave} isLoading={isSaving} size="sm">
          Continue
        </Button>,
        <Button key="download" variant="secondary" onClick={handleDownload} icon={<DownloadIcon />} size="sm">
          Download
        </Button>,
        <Button key="cancel" variant="link" onClick={onClose} size="sm">
          Cancel
        </Button>,
      ]}
    >
      <Form className="pf-v5-c-form--compact">
        {/* File Upload - Compact */}
        <FormGroup fieldId="prompt-file-upload" style={{ marginBottom: '0.75rem' }}>
          <FileUpload
            id="prompt-file-upload"
            type="text"
            value=""
            filename=""
            filenamePlaceholder="Upload YAML prompt..."
            onFileInputChange={handleFileUpload}
            hideDefaultPreview
            browseButtonText="Upload"
          />
        </FormGroup>

        {/* Compact Info with Example Button */}
        <div style={{ 
          padding: '0.5rem 0.75rem', 
          background: '#e7f1fa', 
          borderRadius: '4px', 
          marginBottom: '0.75rem',
          fontSize: '0.8rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span>
            Use <code>{'{{variable}}'}</code> for dynamic content
          </span>
          <Popover
            headerContent={
              <div>
                <InfoCircleIcon style={{ color: '#0066cc', marginRight: '0.5rem' }} />
                <strong>Example Prompt Template</strong>
              </div>
            }
            bodyContent={
              <div>
                <p style={{ 
                  marginBottom: '1rem', 
                  fontSize: '0.9rem',
                  color: '#6a6e73',
                  borderLeft: '3px solid #0066cc',
                  paddingLeft: '0.75rem',
                  fontStyle: 'italic'
                }}>
                  Example for {bundleType === 'summary_generation' ? 'summary generation' : (bundleType || 'custom').replace('_', ' ')}:
                </p>
                <CodeBlock>
                  <CodeBlockCode style={{ 
                    maxHeight: '300px', 
                    overflow: 'auto',
                    fontSize: '0.8rem',
                  }}>
                    {getExamplePromptContent()}
                  </CodeBlockCode>
                </CodeBlock>
                <p style={{ 
                  marginTop: '0.75rem', 
                  fontSize: '0.85rem',
                  color: '#6a6e73',
                  fontStyle: 'italic'
                }}>
                  Copy this structure and customize it for your needs
                </p>
              </div>
            }
            minWidth="500px"
            maxWidth="600px"
            position="auto"
          >
            <Button
              variant="link"
              icon={<HelpIcon />}
              isSmall
            >
              Example
            </Button>
          </Popover>
        </div>

        {/* Prompt File Name */}
        <FormGroup
          label="Prompt File Name"
          isRequired
          fieldId="prompt-file-name"
          style={{ marginBottom: '0.75rem' }}
        >
          <TextInput
            isRequired
            type="text"
            id="prompt-file-name"
            value={promptFileName}
            onChange={(event, value) => setPromptFileName(value)}
            placeholder="e.g., my_qa_prompt"
          />
        </FormGroup>

        {/* System Message */}
        <FormGroup
          label="System Message"
          isRequired
          fieldId="system-message"
          style={{ marginBottom: '0.75rem' }}
        >
          <TextArea
            isRequired
            id="system-message"
            value={systemMessage}
            onChange={(event, value) => setSystemMessage(value)}
            rows={2}
            placeholder="You are a helpful AI assistant..."
            style={{ resize: 'vertical', fontSize: '12px' }}
          />
        </FormGroup>

        {/* User Message Template */}
        <FormGroup
          label="User Message Template"
          isRequired
          fieldId="user-message"
          style={{ marginBottom: '0.5rem' }}
        >
          <TextArea
            isRequired
            id="user-message"
            value={userMessage}
            onChange={(event, value) => setUserMessage(value)}
            rows={5}
            placeholder="Process the following:\n\n{{document}}"
            style={{ resize: 'vertical', fontSize: '12px' }}
          />
        </FormGroup>

        {/* Compact Variables & Preview */}
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem' }}>
          <ExpandableSection
            toggleText="Variables"
            isExpanded={showVariables}
            onToggle={() => setShowVariables(!showVariables)}
            style={{ flex: 1 }}
          >
            <div style={{ padding: '0.5rem', background: '#f5f5f5', borderRadius: '4px' }}>
              <LabelGroup>
                {availableVars.map(variable => (
                  <Label key={variable} color="blue" isCompact>
                    {`{{${variable}}}`}
                  </Label>
                ))}
              </LabelGroup>
            </div>
          </ExpandableSection>

          <ExpandableSection
            toggleText="Preview YAML"
            isExpanded={showPreview}
            onToggle={() => setShowPreview(!showPreview)}
            style={{ flex: 1 }}
          >
            <CodeBlock>
              <CodeBlockCode style={{ fontSize: '0.7rem' }}>
{`- role: system
  content: ${systemMessage || '...'}

- role: user
  content: |
    ${userMessage || '...'}`}
              </CodeBlockCode>
            </CodeBlock>
          </ExpandableSection>
        </div>
      </Form>
    </Modal>
  );
};

export default PromptEditorModal;

