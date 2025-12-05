# How to Upload Yelp JSON Files to Render

After your app is deployed on Render, follow these steps to upload the large Yelp JSON files:

## Method 1: Using Render Shell (Easiest)

1. **Access Render Shell**
   - Go to your Render dashboard
   - Click on your web service
   - Click on the "Shell" tab
   - This opens a terminal in your deployed instance

2. **Upload Files Using wget/curl**
   
   If you have the files hosted somewhere (Google Drive, Dropbox, etc.):
   ```bash
   # Navigate to project root
   cd /opt/render/project/src
   
   # Download from a URL (if you have direct download links)
   wget -O yelp_academic_dataset_business.json "YOUR_DOWNLOAD_URL"
   wget -O yelp_academic_dataset_review.json "YOUR_DOWNLOAD_URL"
   ```

3. **Upload Files Using SCP (from your local machine)**
   
   ```bash
   # From your local machine, use scp to upload
   # First, get your Render service details from the dashboard
   scp yelp_academic_dataset_business.json user@your-render-instance:/opt/render/project/src/
   scp yelp_academic_dataset_review.json user@your-render-instance:/opt/render/project/src/
   ```

## Method 2: Using Cloud Storage (Recommended for Large Files)

1. **Upload to Cloud Storage First**
   - Upload files to AWS S3, Google Cloud Storage, or similar
   - Make them publicly accessible or use credentials

2. **Download During Build or Runtime**
   - Add environment variables in Render dashboard for cloud storage credentials
   - Modify build command or add a startup script to download files

## Method 3: Using Render Persistent Disk (Paid Plans)

1. **Upgrade to Paid Plan**
   - Render's paid plans support persistent disks
   - Files on persistent disk survive deployments

2. **Attach Disk and Upload**
   - Create persistent disk in Render dashboard
   - Mount it to your service
   - Upload files to the mounted disk

## Quick Check

After uploading, verify files are in place:
```bash
# In Render Shell
cd /opt/render/project/src
ls -lh yelp_academic_dataset_*.json
```

## Important Notes

- **File Locations**: Files should be in the project root directory (same level as `app.py`)
- **File Names**: Must be exactly:
  - `yelp_academic_dataset_business.json`
  - `yelp_academic_dataset_review.json`
- **Restart Service**: After uploading, restart your Render service for changes to take effect
- **Free Tier**: Files uploaded to free tier instances may be lost on redeploy (use persistent disk for paid plans)

## Alternative: Download Script

You can also create a simple Python script to download files during app startup if you host them elsewhere. The app will automatically detect and use the files once they're in place.

