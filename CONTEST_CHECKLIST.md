# Contest checklist

- [x] Source code included
- [x] Dockerfile included
- [x] requirements.txt included
- [x] README with build and run instructions included
- [x] One input case = 4 images jointly processed into one DOE
- [x] Main output file = `doe.png`, PNG, 128x128, 8-bit grayscale
- [x] Additional outputs = 4 reconstructed intensities + SSIM/Pearson metrics
- [x] No pretrained weights are used; model is initialized from scratch for each case
- [x] Maximum optimization steps per case = 100
