# Troubleshooting Guide

## Common Issues and Solutions

### 1. Streamlit Crashes with "GGML_ASSERT: n_threads > 0"

**Symptoms:**
- Streamlit starts but crashes when trying to load the Mistral model
- Error message: `GGML_ASSERT: C:\...\llama.cpp:14470: n_threads > 0`

**Solution:**
```bash
# Option 1: Run the fix script
python fix_thread_issue.py

# Option 2: Use basic mode (no ML dependencies)
python launch_basic.py
```

### 2. Windows HuggingFace Hub Symlink Warnings

**Symptoms:**
- Warning: `huggingface_hub cache-system uses symlinks by default...`
- Warning: `your machine does not support them in C:\Users\...\models--sentence-transformers--all-MiniLM-L6-v2`
- Streamlit loads but shows warnings when loading all-MiniLM-L6-v2 model

**Solutions:**

#### Quick Fix - Environment Variable:
```bash
# Set environment variable to disable warning
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Or run the fix script
python fix_windows_compatibility.py
```

#### Permanent Fix - Enable Developer Mode:
1. Open Windows Settings (Windows + I)
2. Go to Update & Security → For developers
3. Select "Developer mode"
4. Restart your computer

#### Alternative - Run as Administrator:
- Right-click Command Prompt → "Run as administrator"
- Launch Streamlit from the admin prompt

### 3. Streamlit Crashes During Model Loading

**Symptoms:**
- Streamlit loads but crashes when downloading/loading sentence-transformers
- Warning messages about `resume_download` or `torch.utils._pytree`
- Memory errors or hanging during model initialization

**Solutions:**

#### Quick Fix - Use Basic Mode:
```bash
# Install only basic requirements
pip install -r requirements_basic_only.txt

# Launch basic dashboard
python launch_basic.py
# OR
streamlit run streamlit_basic_dashboard.py
```

#### Full Fix - Install All Dependencies:
```bash
# Install complete requirements
pip install -r requirements_complete.txt

# Use safe mode launcher
streamlit run launch_streamlit_safe.py
```

### 3. Memory Issues

**Symptoms:**
- "Out of memory" errors
- System becomes unresponsive
- Streamlit crashes without error message

**Solutions:**
1. **Close other applications** to free up RAM
2. **Use basic mode** (requires only ~1GB RAM vs 8GB+ for full mode)
3. **Reduce model parameters** in config:
   ```python
   config.model.n_ctx = 1024  # Reduce from 4096
   config.model.max_tokens = 128  # Reduce from 512
   ```

### 4. Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'xyz'`
- Import errors when starting Streamlit

**Solutions:**
```bash
# For basic functionality only
pip install -r requirements_basic_only.txt

# For full functionality
pip install -r requirements_complete.txt

# If using conda
conda install pandas numpy streamlit
```

### 5. Model File Not Found

**Symptoms:**
- "Model file not found" error
- Dashboard shows "❌ Mistral Model" in sidebar

**Solution:**
```bash
# Download the model file
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Verify the file exists
ls -la mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 6. Slow Performance

**Symptoms:**
- Very slow model loading
- Long response times
- UI becomes unresponsive

**Solutions:**
1. **Use basic mode** for faster analysis without AI inference
2. **Reduce model parameters**:
   - Lower `n_ctx` (context window)
   - Reduce `max_tokens`
   - Use fewer threads
3. **Check system resources**:
   ```bash
   # Check available memory
   python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024**3:.1f}GB available')"
   
   # Check CPU usage
   python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}% usage')"
   ```

## Mode Comparison

### Basic Mode (Recommended for Testing)
- **Launch**: `python launch_basic.py`
- **Requirements**: pandas, numpy, streamlit (~100MB)
- **RAM Usage**: ~1GB
- **Features**: Data processing, basic quality analysis, simple sentiment analysis
- **Limitations**: No AI inference, no RAG, no DeepEval

### Full Mode (Production)
- **Launch**: `streamlit run enhanced_streamlit_dashboard.py`
- **Requirements**: All ML dependencies (~2GB download)
- **RAM Usage**: 8GB+
- **Features**: Complete AI evaluation, RAG pipeline, DeepEval metrics
- **Requirements**: Mistral model file (4.1GB)

### Safe Mode (Diagnostic)
- **Launch**: `streamlit run launch_streamlit_safe.py`
- **Purpose**: System diagnostics and safe configuration
- **Features**: System checks, safe model loading, troubleshooting

## Step-by-Step Recovery

If you're having issues, follow these steps in order:

### Step 1: Basic Functionality Test
```bash
# Test basic Python imports
python -c "import pandas, numpy, streamlit; print('✅ Basic imports OK')"

# Test data processor
python -c "from data_processor import CallTranscriptProcessor; print('✅ Data processor OK')"
```

### Step 2: Launch Basic Mode
```bash
# Install minimal requirements
pip install pandas numpy streamlit

# Launch basic dashboard
python launch_basic.py
```

### Step 3: Test with Sample Data
1. Upload the provided "Call Transcript Sample 1.json"
2. Verify basic analysis works
3. Check results in the Results tab

### Step 4: Upgrade to Full Mode (Optional)
```bash
# Install full requirements
pip install -r requirements_complete.txt

# Download model
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Test safe mode first
streamlit run launch_streamlit_safe.py
```

## Getting Help

### System Information
Run this to get system info for troubleshooting:
```bash
python -c "
import sys, platform, psutil
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'RAM: {psutil.virtual_memory().total/1024**3:.1f}GB total')
print(f'CPU Cores: {psutil.cpu_count()}')
"
```

### Log Files
Check these locations for error logs:
- Streamlit logs: Usually displayed in terminal
- Application logs: `logs/` directory (if created)
- System logs: Check Windows Event Viewer or system logs

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `GGML_ASSERT: n_threads > 0` | Thread configuration issue | Run `python fix_thread_issue.py` |
| `ModuleNotFoundError` | Missing dependencies | Install requirements |
| `Model file not found` | Missing Mistral model | Download model file |
| `Out of memory` | Insufficient RAM | Use basic mode or close applications |
| `torch.utils._pytree` warning | Dependency version conflict | Ignore warning or update dependencies |

## Contact Support

If you're still having issues:
1. Try basic mode first: `python launch_basic.py`
2. Check the system requirements
3. Provide system information and error logs
4. Specify which mode you're trying to use

The basic mode should work on most systems and provides core functionality for transcript analysis without the heavy ML dependencies.
