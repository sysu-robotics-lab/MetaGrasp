gdown https://drive.google.com/uc?id=1BCJkYBIA1wnPtmEZ_MCA9HMaZ6J6DlUg 

tar -zxvf metagrasp.tar.gz 
rm metagrasp.tar.gz

cd metagrasp

mv ur.ttt ../data_collection/

tar -zxvf 3dnet.tar.gz
rm 3dnet.tar.gz
mv 3dnet ../data_collection/

tar -zxvf model.tar.gz
rm model.tar.gz
mv metagrasp ../models/

cd ..
rm -rf metagrasp
