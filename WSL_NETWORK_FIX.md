# WSL Network Error Fix - Step by Step

## Your Current Error

```
[Errno 101] Network is unreachable' thrown while requesting HEAD https://huggingface.co/...
```

**Translation:** Your WSL cannot reach the internet to download model metadata.

---

## ✅ Recommended Fix: Use Your Downloaded Models

You already have models on D: drive. Let's use them instead of downloading.

### Step 1: Find Your Model Location

```bash
# In WSL terminal (where you are now):
find /mnt/d -name "*Mixtral*" -type d 2>/dev/null
```

**Expected output:** Something like:
```
/mnt/d/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1
```

### Step 2: Find the Snapshot Directory

```bash
# Copy the path from Step 1 and run:
ls /mnt/d/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/
```

**Expected output:** A long hash like `d3c35786dd5f30f9326bf61fd29df99f7ce1b02d`

### Step 3: Use Full Path in Command

```bash
# Replace <hash> with the actual hash from Step 2
python3 main.py \
  --model /mnt/d/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/<hash> \
  --prompt "Hello" \
  --local-files-only \
  --max-new-tokens 32
```

**Example (with actual hash):**
```bash
python3 main.py \
  --model /mnt/d/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/d3c35786dd5f30f9326bf61fd29df99f7ce1b02d \
  --prompt "Hello" \
  --local-files-only \
  --max-new-tokens 32
```

**Key flag:** `--local-files-only` tells it to NEVER try downloading.

---

## Alternative: Set Environment Variables

If you want to keep using model IDs like `mistralai/Mixtral-8x7B-Instruct-v0.1`:

### Step 1: Find Your HF_HOME

```bash
# In Windows Command Prompt (open new window):
echo %HF_HOME%
```

If it shows `D:\models` or similar, continue to Step 2.

### Step 2: Export in WSL

```bash
# In WSL (your current terminal):
export HF_HOME=/mnt/d/models  # Adjust to match YOUR path
export TRANSFORMERS_CACHE=/mnt/d/models
export HF_DATASETS_CACHE=/mnt/d/models

# Verify it's set:
echo $HF_HOME
```

### Step 3: Run Command

```bash
python3 main.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --prompt "Hello" \
  --local-files-only
```

### Step 4: Make It Permanent (Optional)

```bash
# Add to WSL shell config
echo 'export HF_HOME=/mnt/d/models' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/d/models' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/mnt/d/models' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

---

## If You Need to Fix WSL Networking (Advanced)

**Only do this if you need to download NEW models.**

### Quick Test

```bash
# Should return HTTP 200:
curl -I -m 5 https://huggingface.co
```

If timeout/error, fix networking:

### Fix Method 1: DNS Reset

```bash
# In WSL:
sudo rm /etc/resolv.conf
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 8.8.4.4" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf

# Test:
curl -I https://huggingface.co
```

### Fix Method 2: Windows Network Reset

**Run in Windows PowerShell as Administrator:**

```powershell
wsl --shutdown
netsh winsock reset
netsh int ip reset all
netsh winhttp reset proxy
ipconfig /flushdns

# Restart computer (recommended)
# Or restart WSL:
wsl
```

### Fix Method 3: WSL Config

**Create/edit `/etc/wsl.conf` in WSL:**

```bash
sudo nano /etc/wsl.conf
```

**Add:**
```ini
[network]
generateResolvConf = false
```

**Save (Ctrl+O, Enter, Ctrl+X), then:**

```bash
exit  # Exit WSL
wsl --shutdown  # From Windows
wsl  # Restart

# Set DNS manually:
sudo rm /etc/resolv.conf
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 8.8.4.4" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf
```

---

## Quick Reference: What to Do NOW

1. **Find your model:**
   ```bash
   find /mnt/d -name "*Mixtral*" -type d 2>/dev/null
   ```

2. **Get snapshot hash:**
   ```bash
   ls <path-from-step-1>/snapshots/
   ```

3. **Run with full path:**
   ```bash
   python3 main.py \
     --model <path>/snapshots/<hash> \
     --prompt "Hello" \
     --local-files-only
   ```

**Done!** No network needed.

---

## Summary

**Root Problem:** WSL can't reach internet → can't download model config

**Solutions:**
1. ✅ **Use local path** (recommended, works immediately)
2. ✅ **Set HF_HOME** (cleaner commands, still local)
3. ⚠️ **Fix networking** (only if downloading new models)

**Your Setup:**
- Models: D: drive (Windows) = `/mnt/d/...` (WSL)
- Code: C: drive (Windows) = `/mnt/c/...` (WSL)
- Virtual env: WSL (`env`)

**Why This Happened:**
- You set `HF_HOME` in Windows (PowerShell/CMD)
- But WSL has its own environment variables
- WSL tried to download because it didn't know about D: drive cache

**Prevention:**
- Always export `HF_HOME` in WSL when using venv
- Or use `--local-files-only` flag
- Or use full model paths

---

**Questions?** Read [QUICK_START_WSL.md](QUICK_START_WSL.md) for complete WSL setup.
