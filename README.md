# dec3b9de2b9176d4573f98dd71f1d3d75a7fde3d1820909422254277b0f2dfa7

```bash
git clone --recursive https://github.com/naver/dust3r
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ./code/checkpoints/
docker compose -f docker-compose-cuda.yml up --build -d
```
