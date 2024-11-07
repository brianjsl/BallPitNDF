if [ -z $CKPT_DIR ]; then echo 'Please define $CKPT_DIR first'
else
mkdir $CKPT_DIR
wget -0 $CKPT_DIR/lndf_weights.pth https://www.dropbox.com/s/mtni5sh01dxxjs7/lndf_weights.pth?dl=0
wget -O $CKPT_DIR/model_weights/lndf_no_se3_weights.pth https://www.dropbox.com/s/mqb28hxo0m2r2a5/lndf_no_se3_weights?dl=0
wget -O $CKPT_DIR/model_weights/ndf_weights.pth https://www.dropbox.com/s/hm4hty56ldu1wb5/multi_category_weights.pth?dl=0
fi