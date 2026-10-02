# Background Remover Frontend - Vercel Deployment

This is the frontend UI for the Background Remover application, designed to be deployed on Vercel.

## Features

- Upload images from your computer
- Process images from URLs
- Real-time preview
- Download processed images
- Beautiful, responsive UI
- Instant background removal using AI

## Deployment on Vercel

### Step 1: Connect to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub/GitLab/Bitbucket
3. Click "New Project"
4. Select your repository
5. Click "Deploy"

### Step 2: Configure Environment

1. No environment variables needed
2. The frontend connects to the Render API directly
3. Update the `API_URL` in `public/index.html` with your Render API URL

### Step 3: Update API URL

In `public/index.html`, update this line with your Render API URL:

```javascript
const API_URL = 'https://your-render-api-url.onrender.com';
```

Example:
```javascript
const API_URL = 'https://bg-remove-api-yvjw.onrender.com';
```

## How It Works

1. User uploads an image or provides an image URL
2. Frontend sends request to Render API
3. Render API processes the image with rembg
4. API returns processed image as PNG
5. Frontend displays and allows download

## File Structure

```
.
├── public/
│   └── index.html      (Main UI)
├── vercel.json         (Vercel configuration)
├── package.json        (Dependencies)
└── README-VERCEL.md    (This file)
```

## Local Testing

To test locally:

1. Make sure Render API is running
2. Update `API_URL` in `public/index.html` to your local API URL (e.g., `http://localhost:3001`)
3. Open `public/index.html` directly in a browser

Or use a local server:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`

## API Endpoints Used

### POST /api/remove-background

Upload an image file and get background removed.

**Request:**
- Content-Type: multipart/form-data
- Field: image

**Response:**
- Content-Type: image/png
- Body: Processed image

### POST /api/remove-background-url

Process image from a URL.

**Request:**
- Content-Type: application/json
- Body: `{"imageUrl": "https://example.com/image.jpg"}`

**Response:**
- Content-Type: image/png
- Body: Processed image

## Troubleshooting

### "Failed to process image"

- Check if Render API is running
- Check if API_URL is correct
- Check browser console for CORS errors

### "Unable to fetch image from URL"

- The URL must be publicly accessible
- Some URLs may have CORS restrictions

## License

MIT
