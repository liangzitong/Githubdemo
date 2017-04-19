#-*- coding:utf-8 -*-
import re
import requests
MaxSearchPage = 100 # 收索页数
CurrentPage = 0 # 当前正在搜索的页数
DefaultPath = "/Users/gushixin/Desktop/OwnerSensor/data/other/" # 默认储存位置
NeedSave = 1 # 是否需要储存
def imageFiler(content): # 通过正则获取当前页面的图片地址数组
        return re.findall('"objURL":"(.*?)"',content,re.S)
def nextSource(content): # 通过正则获取下一页的网址
        next = re.findall('<div id="page">.*<a href="(.*?)" class="n">',content,re.S)[0]
        print("---------" + "http://image.baidu.com" + next) 
        return next
def spidler(source):
        content = requests.get(source).text  # 通过链接获取内容
        imageArr = imageFiler(content) # 获取图片数组
        global CurrentPage
        print("Current page:" + str(CurrentPage) + "**********************************")
        for imageUrl in imageArr:
            print(imageUrl)
            global  NeedSave
            global MaxSearchPage
            if NeedSave:  # 如果需要保存保存
                global DefaultPath
                try:                
                    picture = requests.get(imageUrl,timeout=20) # 下载图片并设置超时时间,如果图片地址错误就不继续等待了
                except:                
                    print("Download image error! errorUrl:" + imageUrl)                
                    continue            
                pictureSavePath = DefaultPath + imageUrl.replace('/','') # 创建图片保存的路径
                fp = open(pictureSavePath,'wb') # 以写入二进制的方式打开文件            
                fp.write(picture.content)
                fp.close()
                if (imageUrl == imageArr[-1]) and (CurrentPage <= MaxSearchPage):
                    CurrentPage += 1
                    spidler("http://image.baidu.com" + nextSource(content)) # 爬取完毕后通过下一页地址继续爬取
            else:   
                if CurrentPage <= MaxSearchPage:
                    if nextSource(content):
                        CurrentPage += 1                                         
                        spidler("http://image.baidu.com" + nextSource(content)) # 爬取完毕后通过下一页地址继续爬取

def beginSearch(page=100,save=1,savePath="/Users/gushixin/Desktop/OwnerSensor/data/other/"): # (page:爬取页数,save:是否储存,savePath:默认储存路径)
        global MaxSearchPage,NeedSave,DefaultPath
        MaxSearchPage = page
        NeedSave = save
        DefaultPath = savePath
        key = raw_input("Please input you want search: ") 
        StartSource = "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=" + str(key) + "&ct=201326592&v=flip" # 分析链接可以得到,替换其`word`值后面的数据来收索关键词
        spidler(StartSource)
beginSearch(page=50,save=1)