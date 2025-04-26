// Initialize block types from the server
const blockTypes = window.blockTypes;
let blocks = [];
let currentBlockId = null;
let isDraggingBlock = false;
let draggedBlock = null;
let dragOffset = { x: 0, y: 0 };
const blockConfigModal = new bootstrap.Modal(document.getElementById('blockConfigModal'));
const yamlInputModal = new bootstrap.Modal(document.getElementById('yamlInputModal'));

// Make blocks draggable
document.querySelectorAll('.block[data-block-type]').forEach(block => {
    block.setAttribute('draggable', 'true');
    block.addEventListener('dragstart', handleDragStart);
});

const flowCanvas = document.getElementById('flow-canvas');
flowCanvas.addEventListener('dragover', handleDragOver);
flowCanvas.addEventListener('dragenter', handleDragEnter);
flowCanvas.addEventListener('dragleave', handleDragLeave);
flowCanvas.addEventListener('drop', handleDrop);

function handleDragStart(e) {
    e.dataTransfer.setData('text/plain', e.target.dataset.blockType);
    e.dataTransfer.effectAllowed = 'copy';
    e.target.classList.add('dragging');
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

function handleDragEnter(e) {
    e.preventDefault();
    flowCanvas.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    if (!flowCanvas.contains(e.relatedTarget)) {
        flowCanvas.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    flowCanvas.classList.remove('drag-over');
    document.querySelector('.dragging')?.classList.remove('dragging');
    
    const blockType = e.dataTransfer.getData('text/plain');
    const blockInfo = blockTypes[blockType];
    
    // Calculate position relative to the canvas
    const rect = flowCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const block = {
        id: Date.now(),
        type: blockType,
        name: blockInfo.name,
        config: {},
        position: { x, y }
    };
    
    blocks.push(block);
    renderBlocks();
}

function handleBlockDragStart(e) {
    if (isDraggingBlock) return;
    
    isDraggingBlock = true;
    draggedBlock = e.target;
    draggedBlock.classList.add('dragging');
    
    // Set the drag image to be transparent
    const dragImage = document.createElement('div');
    dragImage.style.opacity = '0';
    document.body.appendChild(dragImage);
    e.dataTransfer.setDragImage(dragImage, 0, 0);
    setTimeout(() => document.body.removeChild(dragImage), 0);
}

function handleBlockDragEnd(e) {
    if (!isDraggingBlock || !draggedBlock) return;
    
    isDraggingBlock = false;
    draggedBlock.classList.remove('dragging');
    
    // Hide all drop indicators and remove drag-over classes
    document.querySelectorAll('.drop-indicator').forEach(indicator => {
        indicator.style.display = 'none';
        indicator.classList.remove('active');
    });
    document.querySelectorAll('.block').forEach(block => {
        block.classList.remove('drag-over', 'drag-over-above');
    });
    
    draggedBlock = null;
}

function handleBlockDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleBlockDragEnter(e) {
    e.preventDefault();
    if (!isDraggingBlock || !draggedBlock) return;
    
    const targetBlock = e.target.closest('.block');
    if (!targetBlock || targetBlock === draggedBlock) return;
    
    const rect = targetBlock.getBoundingClientRect();
    const mouseY = e.clientY;
    const threshold = rect.top + rect.height / 2;
    
    // Show appropriate drop indicator and add visual feedback
    const dropIndicatorAbove = targetBlock.querySelector('.drop-indicator.above');
    const dropIndicatorBelow = targetBlock.querySelector('.drop-indicator.below');
    
    if (mouseY < threshold) {
        dropIndicatorAbove.style.display = 'block';
        dropIndicatorAbove.classList.add('active');
        dropIndicatorBelow.style.display = 'none';
        dropIndicatorBelow.classList.remove('active');
        targetBlock.classList.add('drag-over-above');
        targetBlock.classList.remove('drag-over');
    } else {
        dropIndicatorAbove.style.display = 'none';
        dropIndicatorAbove.classList.remove('active');
        dropIndicatorBelow.style.display = 'block';
        dropIndicatorBelow.classList.add('active');
        targetBlock.classList.add('drag-over');
        targetBlock.classList.remove('drag-over-above');
    }
}

function handleBlockDragLeave(e) {
    e.preventDefault();
    const targetBlock = e.target.closest('.block');
    if (!targetBlock) return;
    
    // Hide drop indicators and remove visual feedback
    targetBlock.querySelectorAll('.drop-indicator').forEach(indicator => {
        indicator.style.display = 'none';
        indicator.classList.remove('active');
    });
    targetBlock.classList.remove('drag-over', 'drag-over-above');
}

function handleBlockDrop(e) {
    e.preventDefault();
    if (!isDraggingBlock || !draggedBlock) return;
    
    const targetBlock = e.target.closest('.block');
    if (!targetBlock || targetBlock === draggedBlock) return;
    
    const draggedIndex = parseInt(draggedBlock.dataset.index);
    const targetIndex = parseInt(targetBlock.dataset.index);
    
    // Get the ordered blocks
    const orderedBlocks = getOrderedBlocks();
    
    // Remove the dragged block
    const [draggedBlockData] = orderedBlocks.splice(draggedIndex, 1);
    
    // Determine the new position
    const rect = targetBlock.getBoundingClientRect();
    const mouseY = e.clientY;
    const threshold = rect.top + rect.height / 2;
    const newIndex = mouseY < threshold ? targetIndex : targetIndex + 1;
    
    // Insert the block at the new position
    orderedBlocks.splice(newIndex, 0, draggedBlockData);
    
    // Update the blocks array
    blocks = orderedBlocks;
    
    // Hide all drop indicators and remove visual feedback
    document.querySelectorAll('.drop-indicator').forEach(indicator => {
        indicator.style.display = 'none';
        indicator.classList.remove('active');
    });
    document.querySelectorAll('.block').forEach(block => {
        block.classList.remove('drag-over', 'drag-over-above');
    });
    
    // Re-render the blocks with a smooth transition
    renderBlocks();
}

function removeBlock(id) {
    blocks = blocks.filter(block => block.id !== id);
    renderBlocks();
}

function configureBlock(id) {
    currentBlockId = id;
    const block = blocks.find(b => b.id === id);
    if (!block) return;

    const blockInfo = blockTypes[block.type];
    const form = document.getElementById('blockConfigForm');
    form.innerHTML = '';

    // Add block name field
    const nameGroup = document.createElement('div');
    nameGroup.className = 'mb-3';
    
    const nameLabel = document.createElement('label');
    nameLabel.className = 'form-label';
    nameLabel.textContent = 'Block Name';
    nameLabel.innerHTML += ' <span class="text-danger">*</span>';
    
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'form-control';
    nameInput.name = 'block_name';
    nameInput.value = block.config.block_name || `${block.type}_${blocks.indexOf(block)}`;
    nameInput.required = true;
    
    nameGroup.appendChild(nameLabel);
    nameGroup.appendChild(nameInput);
    form.appendChild(nameGroup);

    // Add other configuration fields
    Object.entries(blockInfo.config).forEach(([key, config]) => {
        if (key === 'block_name') return; // Skip block_name as we already added it
        
        const formGroup = document.createElement('div');
        formGroup.className = 'mb-3';
        
        const label = document.createElement('label');
        label.className = 'form-label';
        label.textContent = key;
        if (config.required) {
            label.innerHTML += ' <span class="text-danger">*</span>';
        }
        
        let input;
        if (config.type === 'number') {
            input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control';
            input.value = block.config[key] || config.default || '';
        } else if (config.type === 'array') {
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control';
            input.value = (block.config[key] || []).join(', ');
            input.placeholder = 'Comma-separated values';
        } else if (config.type === 'object') {
            input = document.createElement('textarea');
            input.className = 'form-control';
            input.value = JSON.stringify(block.config[key] || {}, null, 2);
            input.rows = 3;
            input.placeholder = 'JSON object';
        } else if (config.enum) {
            input = document.createElement('select');
            input.className = 'form-select';
            config.enum.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                input.appendChild(optionElement);
            });
            input.value = block.config[key] || '';
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control';
            input.value = block.config[key] || '';
        }
        
        input.name = key;
        input.required = config.required;
        
        formGroup.appendChild(label);
        formGroup.appendChild(input);
        form.appendChild(formGroup);
    });

    blockConfigModal.show();
}

function saveBlockConfig() {
    const form = document.getElementById('blockConfigForm');
    const formData = new FormData(form);
    const config = {};
    
    for (const [key, value] of formData.entries()) {
        if (key === 'block_name') {
            config[key] = value;
        } else {
            const blockInfo = blockTypes[blocks.find(b => b.id === currentBlockId).type];
            const fieldConfig = blockInfo.config[key];
            
            if (fieldConfig.type === 'array') {
                config[key] = value.split(',').map(v => v.trim()).filter(v => v);
            } else if (fieldConfig.type === 'object') {
                try {
                    config[key] = JSON.parse(value);
                } catch (e) {
                    config[key] = {};
                }
            } else if (fieldConfig.type === 'number') {
                config[key] = parseFloat(value);
            } else {
                config[key] = value;
            }
        }
    }
    
    const block = blocks.find(b => b.id === currentBlockId);
    if (block) {
        block.config = config;
    }
    
    blockConfigModal.hide();
    renderBlocks();
}

function saveFlow() {
    const flowData = {
        blocks: blocks.map(block => ({
            type: block.type,
            config: block.config
        }))
    };
    
    // Generate YAML
    fetch('/api/generate_yaml', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(flowData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Create and download YAML file
        const blob = new Blob([data.yaml], { type: 'text/yaml' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'flow.yaml';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error('Error saving flow:', error);
        alert('Error saving flow: ' + error.message);
    });
}

function generateYAML() {
    const flowData = {
        blocks: blocks.map(block => ({
            type: block.type,
            config: block.config
        }))
    };

    fetch('/api/generate_yaml', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(flowData)
    })
    .then(response => response.json())
    .then(data => {
        const yamlOutput = document.getElementById('yaml-output');
        yamlOutput.textContent = data.yaml;
        yamlOutput.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error generating YAML');
    });
}

function loadYAML() {
    yamlInputModal.show();
}

function parseYAML() {
    const yamlContent = document.getElementById('yamlInput').value;
    
    fetch('/api/parse_yaml', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ yaml: yamlContent })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        blocks = data.blocks;
        renderBlocks();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error parsing YAML');
    });
}

function moveBlock(blockId, direction) {
    const currentIndex = blocks.findIndex(b => b.id === blockId);
    if (currentIndex === -1) return;
    
    let newIndex;
    if (direction === 'up') {
        newIndex = currentIndex - 1;
    } else if (direction === 'down') {
        newIndex = currentIndex + 1;
    }
    
    // Check if the new index is valid
    if (newIndex < 0 || newIndex >= blocks.length) {
        return;
    }
    
    // Swap blocks
    [blocks[currentIndex], blocks[newIndex]] = [blocks[newIndex], blocks[currentIndex]];
    renderBlocks();
}

function renderBlocks() {
    flowCanvas.innerHTML = '';
    
    // Render blocks in their current order
    blocks.forEach((block, index) => {
        const blockElement = document.createElement('div');
        blockElement.className = 'block';
        blockElement.dataset.blockId = block.id;
        blockElement.dataset.index = index;
        
        // Add drag handle
        const dragHandle = document.createElement('div');
        dragHandle.className = 'block-handle';
        dragHandle.innerHTML = '<i class="fas fa-grip-vertical"></i>';
        blockElement.appendChild(dragHandle);
        
        const blockContent = document.createElement('div');
        blockContent.className = 'block-content';
        
        // Create name span
        const nameSpan = document.createElement('span');
        nameSpan.className = 'block-name';
        
        // Add block type
        const typeSpan = document.createElement('span');
        typeSpan.className = 'block-type';
        typeSpan.textContent = block.type;
        nameSpan.appendChild(typeSpan);
        
        // Add block name if it exists
        if (block.config && block.config.block_name) {
            const nameValueSpan = document.createElement('span');
            nameValueSpan.className = 'block-name-value';
            nameValueSpan.textContent = block.config.block_name;
            nameSpan.appendChild(nameValueSpan);
        }
        
        blockContent.appendChild(nameSpan);
        
        // Create buttons container
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'block-actions';
        
        // Create up button
        const upButton = document.createElement('button');
        upButton.className = 'btn btn-sm btn-outline-secondary';
        upButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
        upButton.onclick = function() { moveBlock(block.id, 'up'); };
        buttonsContainer.appendChild(upButton);
        
        // Create down button
        const downButton = document.createElement('button');
        downButton.className = 'btn btn-sm btn-outline-secondary';
        downButton.innerHTML = '<i class="fas fa-arrow-down"></i>';
        downButton.onclick = function() { moveBlock(block.id, 'down'); };
        buttonsContainer.appendChild(downButton);
        
        // Create configure button
        const configureButton = document.createElement('button');
        configureButton.className = 'btn btn-sm btn-outline-primary';
        configureButton.onclick = function() { configureBlock(block.id); };
        configureButton.innerHTML = '<i class="fas fa-cog"></i>';
        buttonsContainer.appendChild(configureButton);
        
        // Create remove button
        const removeButton = document.createElement('button');
        removeButton.className = 'btn btn-sm btn-outline-danger';
        removeButton.onclick = function() { removeBlock(block.id); };
        removeButton.innerHTML = '<i class="fas fa-times"></i>';
        buttonsContainer.appendChild(removeButton);
        
        blockContent.appendChild(buttonsContainer);
        blockElement.appendChild(blockContent);
        flowCanvas.appendChild(blockElement);
    });
} 