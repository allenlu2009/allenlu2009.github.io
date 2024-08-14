---
title: Excel Link to MySQL
categories:
- AI
tags: [database, machine learning, ML]
typora-root-url: ../../allenlu2009.github.io
---



作為同門 Microsoft 的資料庫產品，Excel操作Access、SQLServer中的數據非常簡單。但如果想在Excel中處理MySQL中的數據呢？MySQL是和 php 珠聯璧合的一個資料庫，網站開發中經常用到，如果Excel也能訪問它就會給工作帶來很多方便。

最直接的應用就是把現有的 excel table 匯出到 MySQL table 避免重新鍵入資料。

另外的應用是 excel 讀出 MySQL table.  例如，MySQL資料庫中的用戶登錄數據表，有時希望在Excel中對該表進行一些分析：

原文網址：

https://kknews.cc/code/pqbjrv8.html

https://kknews.cc/code/pqbjrv8.html

https://www.youtube.com/watch?v=qK9gPEF606U&ab_channel=SyntaxByte (MySQL -> Excel, very good YouTube)





<img src="/media/image-20211023184418437.png" alt="image-20211023184418437" style="zoom:50%;" />



## 從 MySQL 匯出到 Excel (比較容易)

具體操作步驟：

#### Step 1: 到 MySQL website download ODBC connector driver and install

https://dev.mysql.com/downloads/connector/odbc/

重點是確認 32-bit or 64-bit version.  不過 Windows 10 之後 Excel 都是 64-bit version.  好像也不是什麽問題。

After installation,  go to command window and search odbc

<img src="/media/image-20211023223106919.png" alt="image-20211023223106919" style="zoom: 50%;" />

Then,  add MySQL ODBC Unicode Driver

<img src="/media/image-20211023223348449.png" alt="image-20211023223348449" style="zoom:50%;" />

Then setup the MySQL connection and test it.

<img src="/media/image-20211023223552099.png" alt="image-20211023223552099" style="zoom:50%;" />



此時 ODBC 多了 MySQL local! 

<img src="/media/image-20211023223655068.png" alt="image-20211023223655068" style="zoom:50%;" />





#### Step 2: 到 Excel build the MySQL database link

Now we go to Excel 

Data tab -> Get Data -> from other sources -> from ODBC

<img src="/media/image-20211023223950603.png" alt="image-20211023223950603" style="zoom:50%;" />

Choose MySQL local

<img src="/media/image-20211023224114030.png" alt="image-20211023224114030" style="zoom:50%;" />



下一步是關鍵!  Choose Default or Custom!  and leave blank!

<img src="/media/image-20211023224313640.png" alt="image-20211023224313640" style="zoom:50%;" />



就會出現 MySQL 目前的所有 database in Navigator window

<img src="/media/image-20211023224433565.png" alt="image-20211023224433565" style="zoom:50%;" />

可以直接 navigate the database and **load the table**



<img src="/media/image-20211023224732977.png" alt="image-20211023224732977" style="zoom:50%;" />



或是 **Transform the data** if needed.



## 從 Excel 匯入到 MySQL (比較複雜)

- Local load csv file
- Use Python (or PHP) to insert csv file to MySQL
- Use Excel ODBC ?  (no direct way?)

| Method          | Local or Remote | Format |
| --------------- | --------------- | ------ |
| Manual          | Local load      | csv    |
| Python (or PHP) | Remote          | csv    |
| Commercial tool | Remote          | excel  |



#### Method 0: Use 3rd party tool

Ex1: Navicat  https://www.gushiciku.cn/pl/gQIF/zh-tw => 非常貴！



#### Method 1: Local load csv file

https://chartio.com/resources/tutorials/excel-to-mysql/

1. Download the boats.xlsx file, open in excel, and save as (windows) csv file.

2. Log into the MySQL shell and create a database.  For this example the database is named boatdb.  Not that the --local-infile option is needed by some version of MySQL for the data loading.

   1.  $ mysql -u root -p --local-infile

   2. mysql> create database boatdb;

   3. mysql> use boatdb;

   4. Then define the schema for our boat table using the CREATE TABLE

      ```mysql
      CREATE TABLE boats (
      id INT NOT NULL PRIMARY KEY,
      name VARCHAR(40),
      type VARCHAR(10),
      owner_id INT NOT NULL,
      date_made DATE,
      rental_price FLOAT
      );
      ```



5. 檢查是否 create database and table ok.

   1. $ mysql> show tables;

      <img src="/media/image-20211030230631503.png" alt="image-20211030230631503" style="zoom: 67%;" />

6. 再來就是最關鍵的部分，LOAD DATA command.

```mysql
LOAD DATA LOCAL INFILE "c:/Users/allen/Downloads/boats.csv" INTO TABLE boatdb.boats
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(id, name, type, owner_id, @datevar, rental_price)
set date_made = STR_TO_DATE(@datevar,'%m/%d/%Y');
```

結果失敗了！應該是 permission 的問題。

**ERROR 2068 (HY000): LOAD DATA LOCAL INFILE file request rejected due to restrictions on access.**

我費了九牛二虎之力都無法解決在 sql shell LOAD DATA LOCAL INFILE 的問題！

最後我找到 solution，就是直接在 workbench 的 command window 就可以！

<img src="/media/image-20211031001341480.png" alt="image-20211031001341480" style="zoom:80%;" />

接下來就可以做各種 query!



#### Method 2: Python or PHP

之前 MySQL 的管理 SW 最普遍的是 phpMyAdmin, 主要是用 PHP script 作爲 front-end interface (html) 和 backend database 溝通的工具。 

之後 MySQL 提供的 workbench, 是由 Python 寫的。很自然也是用 Python 作爲和 database 溝通的工具。 

以下是兩者的比較：

**PHPMyAdmin (PHP)**

Pros
* Commonly installed on managed hosting environments
* Web Based which means you can access from any computer
* Local resources aren't used when connecting
* Simplicity

Cons
* No schema visualization
* If remote database working offline can be more difficult

**MySQL Workbench (Python)**

Pros
* Saved SQL statements
* Offline access to remote DB's
* Handle/Store multiple connections in one location

Cons
* Resource consumption
* More complex than the average user would need



參考 [cheahHowUse2019] and [projectproHowConnect2020]，此處我們用 Python connector access the database.   這裏還有一些設定上的插曲。Windows 的 Workbench install Python connector 不是很順利，最後是用 Windows anaconda create a virtual environment，再 install python connection.  嚴格來説和 workbench 沒有一毛錢的關係。 

```shell
conda create --name sql python=3.7  
conda activate sql
conda install -c anaconda mysql-connector-python
```



程式碼分爲三部分

1.  Read CSV file using pandas.
2. Create a database.
3. Create a table and insert csv records to the table.



```python
### Read CSV file

import pandas as pd
empdata = pd.read_csv('C:\\Users\\allen\\Downloads\\empdata.csv', index_col=False, delimiter = ',')
empdata.head()


### Create a database

#import mysql.connector as mysql
#from mysql.connector import Error
#try:

#    conn = msql.connect(host='localhost', user='root',  

#                        password='alu1234') #give ur username, password

#    if conn.is_connected():

#        cursor = conn.cursor()

#        cursor.execute("CREATE DATABASE employee")

#        print("Database is created")

#except Error as e:

#    print("Error while connecting to MySQL", e)


### Insert CSV records to database

import mysql.connector as mysql
from mysql.connector import Error
try:
    conn = mysql.connect(host='localhost', database='employee', user='root', password='alu1234')
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute('DROP TABLE IF EXISTS employee_data;')
        print('Creating table....')

# in the below line please pass the create table statement which you want #to create

        cursor.execute("CREATE TABLE employee_data(first_name varchar(255),last_name varchar(255), \
    	company_name varchar(255),address varchar(255),city varchar(255),county varchar(255), \
    	state varchar(255),zip int,phone1 varchar(255),phone2 varchar(255),email varchar(255), \
    	web varchar(255))")
        print("Table is created....")
        #loop through the data frame
        for i,row in empdata.iterrows():
            #here %S means string values 
            sql = "INSERT INTO employee.employee_data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql, tuple(row))
            print("Record inserted")
            # the connection is not auto committed by default, so we must commit to save our changes
            conn.commit()

except Error as e:
            print("Error while connecting to MySQL", e)


```



使用 Python 還有一個潛在的好處，就是可以用來做 ML/AI 分析。此處不再贅述。



## Reference