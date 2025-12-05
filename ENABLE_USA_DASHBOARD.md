# 🚀 Enable USA Dashboard - Step by Step Guide

## Problem
The USA Dashboard requires Yelp JSON files (~5.3GB) which exceed Hugging Face Spaces' 1GB storage limit.

## Solution: Use Hugging Face Datasets

Upload the Yelp files to a **Hugging Face Dataset** (separate from Spaces), then load them at runtime.

### Step 1: Create a Hugging Face Dataset

1. Go to: https://huggingface.co/datasets
2. Click **"New dataset"**
3. Fill in:
   - **Name**: `yelp-academic-dataset` (or your choice)
   - **Visibility**: Private (recommended) or Public
4. Click **"Create dataset"**

### Step 2: Upload Yelp JSON Files

1. In your dataset repository, go to **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload:
   - `yelp_academic_dataset_review.json` (~5.3GB)
   - `yelp_academic_dataset_business.json` (~118MB)
4. Wait for upload to complete (may take time for large file)

### Step 3: Set Environment Variable in Your Space

1. Go to your Space: https://huggingface.co/spaces/kobrakai11/BetterBites
2. Go to **Settings** → **Variables** tab
3. Add new variable:
   - **Key**: `YELP_DATASET_NAME`
   - **Value**: `yourusername/yelp-academic-dataset` (your dataset name)
4. Click **"Save"**

### Step 4: Update Code (Already Done!)

The code has been updated to:
- ✅ Check for files locally first
- ✅ Download from Hugging Face Datasets if missing
- ✅ Handle errors gracefully

### Step 5: Restart Your Space

1. Go to your Space → **Settings**
2. Click **"Restart this Space"**
3. Wait for rebuild (~5-10 minutes)
4. Check logs to see if files are downloading

### Step 6: Test USA Dashboard

1. Go to your Space → **App** tab
2. Click **"USA"** tab in the dashboard
3. Should now work! 🎉

## Alternative: Direct URL Download

If you have the files hosted elsewhere (Google Drive, Dropbox, etc.):

1. Get a direct download URL
2. Set environment variable:
   - **Key**: `YELP_DATASET_URL`
   - **Value**: Your direct download URL
3. Restart Space

## Troubleshooting

### "Dataset not found"
- Check dataset name is correct: `username/dataset-name`
- Verify dataset is accessible (public or you have access)
- Check environment variable is set correctly

### "Download failed"
- Check internet connection
- Verify URL is accessible
- Check file permissions

### "Still showing error"
- Check Space logs for specific error messages
- Verify files are in the dataset
- Try restarting the Space

---

**Your Space**: https://huggingface.co/spaces/kobrakai11/BetterBites

