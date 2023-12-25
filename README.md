
## NXP AI DEMO

---


### NXP i.MX8M Plus Platform

使用情境 : NXP i.MX8M Plus Platform with VerSlicon Vivante NPU [官網](https://www.verisilicon.com/en/IPPortfolio/VivanteNPUIP)

* 用法一 (影像輸入) : python3 app.py --display 1 --delegate "vx"

* 用法二 (攝鏡頭輸入)  : python3 app.py --display 1 --save 0 --camera 1 --delegate "vx"

</br>

### NXP i.MX 93 Platform

使用情境 : NXP i.MX 93 Platform with ARM NPU [官網](https://www.arm.com/zh-TW/products/silicon-ip-cpu/ethos/ethos-u65)

* 用法一 (影像輸入) : python3 app.py --display 1 --delegate "ethous"

* 用法二 (攝鏡頭輸入)  : python3 app.py --display 1 --save 0 --camera 1 --delegate "ethous"

</br>

-----------------------------------------------------------------------------------------------------------------------------------------------------------

### Release Note
(1) Yocto BSP Base on Version L6.1.36.1.0.0. </br>
(2) YOLOv8 series can't inference by using ethos-U . </br>
(3) YOLOv5 series can't inference by using VIP8000 , but BSP 5.x had work. </br>

-----------------------------------------------------------------------------------------------------------------------------------------------------------

![image](https://github.com/weilly0912/Python-TensorflowLite-DEMO/blob/v7.0/OP-Killer%20%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%87%89%E7%94%A8%E7%A4%BA%E6%84%8F%E5%9C%96.png)
-----------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://github.com/weilly0912/Python-TensorflowLite-DEMO/blob/v7.0/OP-Killer%20%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E6%9B%B4%E5%A4%9A%E5%AF%A6%E9%9A%9B%E6%87%89%E7%94%A8%E7%A4%BA%E6%84%8F%E5%9C%96.png)
-----------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://github.com/weilly0912/Python-TensorflowLite-DEMO/blob/v7.0/OP-Killer%20%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E6%9B%B4%E5%A4%9A%E5%AF%A6%E9%9A%9B%E6%87%89%E7%94%A8%E6%95%B8%E6%93%9A%E8%A1%A8.png)
