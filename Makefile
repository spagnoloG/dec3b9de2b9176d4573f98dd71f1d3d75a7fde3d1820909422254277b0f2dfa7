push_code_vicos:
	scp -r ./code/*py vicos:/home/gasper/res_dust_dir/code/
	scp -r ./code/conf_vicos.yml vicos:/home/gasper/res_dust_dir/code/conf.yml
	ssh vicos "find /home/gasper/res_dust_dir/code -type f -iname '*py' -exec sed -i 's|/dust3r|../dust3r|g' {} +"

pusch_checkpoint_vicos:
	scp ./code/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth vicos:/home/gasper/res_dust_dir/code/checkpoints/

format:
	black ./code/*py
