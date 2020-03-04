# -*- coding: UTF-8 -*-
import tushare as ts
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates


def drawline(tcode, sdate, edate):
    datadf = ts.pro_bar(#pro_api=api,
                        ts_code=tcode,
                        start_date=sdate,
                        end_date=edate,
                        # 前复权处理
                        adj='qfq')

    stockline = [datetime.strptime(d, '%Y%m%d').date() for d in datadf.trade_date]
    plt.plot(stockline, datadf.close, '-', label=datadf.ts_code[0])


def drawmain():
    # 设置时间按“年月”的格式显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
    # X轴按年进行标记，还可以用MonthLocator()和DayLocator()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator()),
    # 自动旋转日期标记以避免重叠
    plt.gcf().autofmt_xdate()
    # 显示图例
    plt.legend()
    # 显示图片
    plt.show()


def taskmain():
    for cd in tscode:
        drawline(cd, startdate, enddate)


# 在tushare官网注册后，进入个人中心得到你的唯一指定token，替换***
ts.set_token('1191f35cb991ffb4250b44e9cb76fafc1533158dda1dd634b51f3822')
# 初始化api
api = ts.pro_api()

# 指定起止日期
startdate = '2015-01-01'
enddate = '2020-02-01'
# 指定股票代码
tscode = {'000333.SZ', '000651.SZ'}
# 主程序
taskmain()
drawmain()
