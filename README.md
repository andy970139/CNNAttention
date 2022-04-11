# CNNAttention
一個使用注意力機制(Attention)的CNN模型實作練習。
首先將注意力機制使用在channel上(參考至Squeeze-and-Excitation Networks此篇論文)，建立名為SEblock的區塊(名稱參考同篇論文)。
由於此結構專注於通道上的注意力(如RGB圖像就是在RGB三個通道上的注意力，隱藏層中就是Feature Map的數量)；概念如圖![image](https://github.com/andy970139/CNNAttention/blob/main/channel%20attention%E7%A4%BA%E6%84%8F%E5%9C%96.png)



先使用全局平均池化層(Fsq)平均通道的特徵圖並壓縮資訊量(亦即(Batch,Channel,w,h)--> (Batch,Channel,1,1))，
再透過兩層全連結層訓練attention權重矩陣(第一個全連結層Fex計算各通道的"非線性"關係，將各通道壓縮成單通道達到混和權重的目的，這邊使用的是Relu函數)，
再透過Fscale還原成原通道數得到attention權重矩陣(這邊使用softmax函數取各通道的注意力比重，這邊與原始論文sigmoid不一樣，希望能比較接近attention的邏輯:各通道的"機率比重")，再將原始圖乘上attention權重矩陣就是注意力機制加乘的結果。 到這裡Squeeze-and-Excitation Networks基本以實作完成。


但是有了channel的注意力肯定是不夠的，因此後來加上了對於特徵圖的注意力機制(w,h上的pixel注意力)，與channel attention不同的是這邊基於特徵圖進行注意力權重計算，將各通道的特徵圖進行平均
，單考慮特徵圖的注意力機制(亦即(Batch,Channel,w,h)--> (Batch,1,w,h))，概念如圖![image](https://github.com/andy970139/CNNAttention/blob/main/spatial%E7%A4%BA%E6%84%8F%E5%9C%96.jpg)



這邊結構參考至CBAM: Convolutional Block Attention Module這篇論文，不過為了簡化這邊一樣只使用平均層(對於特徵圖)，再通過一個7x7的捲積層而非全連結層(這邊參考CBAM的做法，應該是為了保留圖形的結構姓)，
而後面同樣的我使用了softmax函數進行激發(概念上是對於特徵圖的pixel進行"機率權重")，再與原始各通道特徵圖相乘，取得特徵圖注意力加權結果。

總結模型通過普通的CNN層再通過channel attention區塊最後通過spatial attention區塊(特徵圖注意力)，即完成了通道與特徵圖的注意力機制(後面再根據模型目標串連其他網路層，程式碼以全連結層當作範例)
通道注意力與特徵圖注意力的組合方式同樣參考CBAM，論文中作者有分別實驗兩種區塊平行與兩種區塊串聯(順序)進行實驗測試，因此這邊暫時參考這樣的結構。





