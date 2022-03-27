wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=18020270@vnu.edu.vn&password=Luongdangvp2000@&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

unzip gtFine_trainvaltest.zip
mv gtFine_trainvaltest/* data/cityscapes/
rm -d gtFine_trainvaltest
rm gtFine_trainvaltest.zip

unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit_trainvaltest/* data/cityscapes/
rm -d leftImg8bit_trainvaltest
rm leftImg8bit_trainvaltest.zip