# 🚀 Deploy to Hugging Face Spaces

## Quick Start (5 minutes)

### 1. Create Account
- Go to https://huggingface.co/join
- Sign up (free, no credit card)

### 2. Create Space
- Go to https://huggingface.co/spaces
- Click **"New Space"**
- Fill in:
  - **Name**: `betterbites`
  - **SDK**: **Docker** ⚠️ (Select Docker, not Gradio!)
  - **Visibility**: Public
- Click **"Create Space"**

### 3. Connect Repository
- In Space → **Settings** → **Repository**
- Click **"Connect Repository"**
- Select: `Faraj11/CTP-Fall-2025-Project`
- **Branch**: `main`
- **Root directory**: `CTP-Fall-2025-Project-main` ⚠️ **CRITICAL**
- Click **"Save"**

### 4. Deploy
- Build starts automatically
- First build: **10-15 minutes** (downloads BLIP model)
- Monitor progress in build logs
- Wait for "Running" status

### 5. Access
- Your app: `https://yourusername-betterbites.hf.space`
- Share the URL!

## ⚠️ Critical Settings

**Root Directory**: `CTP-Fall-2025-Project-main`
- **MUST** be set in Space settings
- Without this, Dockerfile won't be found

**SDK**: Docker
- **MUST** select "Docker" not "Gradio"
- Your app uses Flask

## What's Included

✅ All Python code  
✅ Flask application  
✅ HTML templates  
✅ Restaurant data (CSV)  
✅ Dockerfile  
✅ Requirements  

## First Use

- **First image search**: 30-45 seconds (model download)
- **Subsequent searches**: 2-5 seconds
- **App sleep**: After 48h inactivity (wakes automatically)

## Troubleshooting

**Build fails?**
- Check root directory is set correctly
- Verify Dockerfile exists
- Review build logs

**App won't start?**
- Check port is 7860
- Verify environment variables
- Review app logs

**Need help?**
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Check build logs for errors

---

**That's it!** Your app will be live in ~15 minutes! 🎉

## Adding Yelp JSON Files Later

Want to enable the USA Dashboard tab? You can add the Yelp JSON files later:

1. Go to your Space → **Files** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload `yelp_academic_dataset_review.json` (~5.3GB) and `yelp_academic_dataset_business.json` (~118MB)
4. Space rebuilds automatically
5. USA Dashboard will work!

**Note**: The app works perfectly without these files - only the USA Dashboard tab requires them. All other features (Text Search, Image Search, NYC Dashboard) work with just the included CSV file.

