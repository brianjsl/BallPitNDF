wget -O lndf_mug_handle_demo.tar.gz https://www.dropbox.com/s/9p286eb5wm9hphu/lndf_mug_handle_demos.tar.gz?dl=0
mv lndf_mug_handle_demo.tar.gz src/demos
tar -xzf lndf_mug_handle_demo.tar.gz
rm lndf_mug_handle_demo.tar.gz

wget -O textures.zip https://www.dropbox.com/scl/fi/7u1yv6zzva4ejv3pzve0a/textures.zip?rlkey=zvohinvnwkskduqoxxjtpc1hs&st=8bshzheo&dl=0
mv textures.zip src/assets/wooden_basket
unzip -xzf textures.zip
rm textures.zip

wget -O src/modules/grasping/lndf_robot/ckpts/lndf_weights.pth https://www.dropbox.com/s/mtni5sh01dxxjs7/lndf_weights.pth?dl=0