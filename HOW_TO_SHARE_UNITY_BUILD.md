# How to Share Unity Build with Friends

## 📍 Your Unity Build Location

Your Unity build is currently located at:
```
C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles\Training\Windows\dp.exe
```

## 📦 What to Share

You need to share the **entire Unity build folder**, not just `dp.exe`. The build typically includes:
- `dp.exe` - Main executable
- `dp_Data/` - Game data folder (textures, models, etc.)
- `MonoBleedingEdge/` - Unity runtime files
- `UnityPlayer.dll` - Unity player library
- Other supporting DLLs and files

**Share the entire `Training/Windows/` folder** (or the parent folder containing it).

## 🚀 Methods to Share

### Method 1: Cloud Storage (Recommended)

**Google Drive / OneDrive / Dropbox:**

1. **Zip the Unity build folder:**
   ```powershell
   # Navigate to the parent folder
   cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
   
   # Create a zip file (Windows 10+)
   Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
   ```

2. **Upload to cloud storage:**
   - Upload `dPickleball_Unity_Build.zip` to Google Drive/OneDrive/Dropbox
   - Share the link with your friend
   - Make sure to set permissions to "Anyone with the link can view" or share directly with your friend

3. **Your friend downloads and extracts:**
   - Download the zip file
   - Extract to a location like `C:\dPickleball\Training\`
   - Update the path in `train.py` to point to `C:\dPickleball\Training\Windows\dp.exe`

### Method 2: File Transfer Service

**WeTransfer / SendAnywhere / Firefox Send:**

1. Zip the Unity build folder (same as Method 1)
2. Go to wetransfer.com or similar service
3. Upload the zip file
4. Send the download link to your friend
5. Link expires after 7 days (WeTransfer) or as specified

### Method 3: GitHub Releases (If Repository is Public)

If your repository is public, you can use GitHub Releases:

1. **Create a release:**
   ```powershell
   # Zip the build (same as Method 1)
   # Then go to GitHub → Your Repository → Releases → Draft a new release
   ```

2. **Upload the zip file as a release asset**
3. **Your friend downloads from the release page**

**Note:** GitHub has a 100MB file size limit for regular uploads. If the build is larger, use Git LFS or a different method.

### Method 4: Direct File Transfer (Same Network)

If you and your friend are on the same network:

1. **Enable file sharing on your computer**
2. **Share the folder** containing the Unity build
3. **Your friend copies it over the network**

## 📋 Quick Steps for You

1. **Navigate to the build folder:**
   ```powershell
   cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
   ```

2. **Create zip file:**
   ```powershell
   Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
   ```

3. **Upload to cloud storage** (Google Drive recommended)

4. **Share the link** with your friend

5. **Tell your friend:**
   - Download and extract the zip
   - Extract to a location like `C:\dPickleball\Training\`
   - Update `UNITY_BUILD_PATH` in `train.py` to: `r"C:\dPickleball\Training\Windows\dp.exe"`

## ✅ Verification

After your friend extracts the build, they should:
1. Navigate to the `Windows` folder
2. Double-click `dp.exe` to verify it runs
3. If the Unity game window opens, the build is correct!

## 📝 Notes

- **File Size:** Unity builds can be 100MB - 2GB+ depending on the game
- **Extraction Time:** Large builds may take several minutes to extract
- **Antivirus:** Some antivirus software may flag `.exe` files - your friend may need to allow it
- **Path Length:** Windows has a 260 character path limit - keep extraction paths short

## 🔗 Alternative: If You Have the Original Source

If you have access to the original Unity project or build source:
- You can rebuild it for your friend
- Or share the original download link if it's still available

