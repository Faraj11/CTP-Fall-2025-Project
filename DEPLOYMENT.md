# Deployment Guide

## Easiest Option: Render (Recommended)

Render is the simplest way to deploy this Flask app. It offers a free tier and automatic deployments from GitHub.

### Step-by-Step Deployment

1. **Sign up for Render**
   - Go to [https://render.com](https://render.com)
   - Sign up with your GitHub account (free)

2. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `Faraj11/CTP-Fall-2025-Project`
   - Select the repository

3. **Configure the Service**
   - **Name**: `betterbites` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or choose a paid plan for better performance)

4. **Environment Variables** (Optional)
   - No environment variables needed for basic deployment
   - The app will work with default settings

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically:
     - Clone your repository
     - Install dependencies
     - Build the app
     - Deploy it
   - First deployment takes 5-10 minutes (includes downloading ML models)

6. **Access Your App**
   - Once deployed, you'll get a URL like: `https://betterbites.onrender.com`
   - The app will be live and accessible!

### Including Large Yelp JSON Files

The Yelp JSON files (~5.3GB total) are too large for GitHub, but you can include them in Render using these methods:

#### Option 1: Upload After Deployment (Easiest)
1. Deploy your app to Render first
2. Go to your Render service dashboard
3. Use Render Shell (available in dashboard) to upload files:
   ```bash
   # In Render Shell, upload files using scp or wget
   # Or use Render's file upload feature if available
   ```

#### Option 2: Use Render Persistent Disk (Paid Plans)
1. Upgrade to a paid Render plan
2. Attach a persistent disk to your service
3. Upload JSON files to the persistent disk
4. Files persist across deployments

#### Option 3: Download During Build (Recommended)
1. Store JSON files in cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Add download script to your build process
3. Files download automatically during deployment

#### Option 4: Git LFS (Limited)
- Use Git Large File Storage
- Still has size limitations but can handle files up to 2GB
- Requires Git LFS setup

**Recommended**: Use Option 3 (cloud storage) for production, or Option 1 for testing.

### Important Notes

- **First Deployment**: The BLIP model (~990MB) will be downloaded during first deployment, so it may take longer
- **Free Tier Limitations**: 
  - Services spin down after 15 minutes of inactivity
  - First request after spin-down may take 30-60 seconds
  - Consider upgrading to a paid plan for production use
- **Data Files**: The `nyc_restaurants_merged.csv` file is included in the repo and will be available
- **Large Yelp Files**: See options above for including JSON files

### Alternative: Railway

Railway is another simple option:

1. Go to [https://railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys
6. Done!

### Alternative: Heroku

1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`
4. Done!

---

**Recommended**: Use **Render** - it's the simplest and most beginner-friendly option!

