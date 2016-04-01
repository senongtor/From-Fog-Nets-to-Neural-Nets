# From-Fog-Nets-to-Neural-Nets
Task:
在submission里的而不在test里的time series的feature是需要第二步测的，看在micro（在周边有）和macro（在时间段内有）在这个时间段内是否有数据
test里在上述时间段内的micro和macro同上
脚本：可以把每个文件的时间段抓出来，上述时间内是否有missing time series
参数：起始时间，结束时间，gap

返回：missing windows
