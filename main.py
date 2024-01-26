import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import wandb
import sklearn
import re
import math
import joblib
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

TACTIC = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 'TA0010', 'TA0040']
TECHNIQUE = ['T1546.010', 'T1205', 'T1546', 'T1189', 'T1553.005', 'T1550', 'T1048', 'T1087.002', 'T1021.001', 'T1598.003', 'T1200', 'T1531', 'T1491.001', 'T1132.001', 'T1055.001', 'T1498.001', 'T1555.005', 'T1102.003', 'T1578.003', 'T1592.002', 'T1090.001', 'T1003.002', 'T1562.002', 'T1619', 'T1021.004', 'T1134.003', 'T1029', 'T1567.002', 'T1561.001', 'T1490', 'T1011.001', 'T1518.001', 'T1210', 'T1497', 'T1072', 'T1134.004', 'T1595.003', 'T1547.012', 'T1498.002', 'T1491', 'T1552.003', 'T1001.002', 'T1585.001', 'T1114', 'T1098.001', 'T1542.003', 'T1622', 'T1563.001', 'T1027.005', 'T1001.001', 'T1495', 'T1505', 'T1546.009', 'T1056.001', 'T1021.003', 'T1104', 'T1041', 'T1548.004', 'T1040', 'T1105', 'T1525', 'T1074.001', 'T1553.006', 'T1213', 'T1547.007', 'T1589.002', 'T1078', 'T1542.005', 'T1053.007', 'T1112', 'T1137.006', 'T1070.006', 'T1114.002', 'T1115', 'T1562.001', 'T1003.008', 'T1561', 'T1535', 'T1621', 'T1546.012', 'T1546.014', 'T1553.002', 'T1591', 'T1578.002', 'T1012', 'T1021', 'T1053.002', 'T1195.003', 'T1548.002', 'T1136.001', 'T1204.001', 'T1137', 'T1132', 'T1564.008', 'T1102.002', 'T1049', 'T1187', 'T1129', 'T1574.012', 'T1070.005', 'T1573', 'T1547.004', 'T1092', 'T1555.004', 'T1037.002', 'T1596.004', 'T1018', 'T1484.002', 'T1055.004', 'T1037', 'T1590.006', 'T1098.005', 'T1052.001', 'T1110.003', 'T1598.002', 'T1564.010', 'T1584.003', 'T1218', 'T1211', 'T1213.003', 'T1590.003', 'T1584.001', 'T1553.001', 'T1550.001', 'T1573.002', 'T1027.004', 'T1542.004', 'T1564.003', 'T1056.004', 'T1584.004', 'T1027.001', 'T1647', 'T1071.001', 'T1218.003', 'T1565.001', 'T1070.004', 'T1596.003', 'T1555.001', 'T1071.004', 'T1114.001', 'T1588.006', 'T1555.003', 'T1055.009', 'T1608.003', 'T1596.005', 'T1102', 'T1583.006', 'T1568.003', 'T1204.002', 'T1053.005', 'T1587.004', 'T1590.001', 'T1574.009', 'T1590.002', 'T1134.002', 'T1098', 'T1574.013', 'T1059.003', 'T1070.002', 'T1110.004', 'T1596.002', 'T1550.003', 'T1608.005', 'T1588.002', 'T1559.003', 'T1489', 'T1574.007', 'T1559.002', 'T1098.002', 'T1030', 'T1574.005', 'T1564.009', 'T1546.006', 'T1563.002', 'T1087.001', 'T1593.001', 'T1087.004', 'T1552.002', 'T1568.001', 'T1047', 'T1020.001', 'T1588.001', 'T1055', 'T1176', 'T1195.001', 'T1496', 'T1055.005', 'T1080', 'T1059.002', 'T1204', 'T1213.001', 'T1566.003', 'T1615', 'T1573.001', 'T1074', 'T1056.003', 'T1562.008', 'T1505.001', 'T1543.003', 'T1202', 'T1595', 'T1480.001', 'T1056.002', 'T1584.005', 'T1218.010', 'T1207', 'T1125', 'T1574.004', 'T1218.004', 'T1127', 'T1547.001', 'T1599', 'T1553', 'T1068', 'T1547.014', 'T1069', 'T1546.005', 'T1566.002', 'T1195.002', 'T1600.001', 'T1218.013', 'T1526', 'T1070.003', 'T1568', 'T1546.004', 'T1556.005', 'T1201', 'T1137.004', 'T1567.001', 'T1048.002', 'T1562.003', 'T1090', 'T1203', 'T1505.005', 'T1484', 'T1059.008', 'T1059.006', 'T1609', 'T1218.012', 'T1611', 'T1558.003', 'T1499', 'T1595.001', 'T1538', 'T1546.011', 'T1499.002', 'T1124', 'T1599.001', 'T1608.001', 'T1027', 'T1534', 'T1110.002', 'T1574.006', 'T1003.004', 'T1053.003', 'T1001', 'T1220', 'T1006', 'T1036.001', 'T1499.003', 'T1055.002', 'T1559', 'T1546.007', 'T1120', 'T1590', 'T1560', 'T1106', 'T1020', 'T1578.001', 'T1594', 'T1585', 'T1595.002', 'T1055.008', 'T1558.002', 'T1499.001', 'T1055.014', 'T1222.002', 'T1574.011', 'T1098.003', 'T1564.001', 'T1055.015', 'T1591.003', 'T1567', 'T1003', 'T1003.005', 'T1566.001', 'T1585.002', 'T1559.001', 'T1219', 'T1114.003', 'T1588.004', 'T1132.002', 'T1587.001', 'T1552.001', 'T1608.002', 'T1546.013', 'T1583.004', 'T1558.001', 'T1602', 'T1547.009', 'T1606.001', 'T1027.006', 'T1003.003', 'T1588.003', 'T1543.002', 'T1102.001', 'T1547.006', 'T1037.005', 'T1123', 'T1039', 'T1530', 'T1592.003', 'T1204.003', 'T1562.007', 'T1556', 'T1574.010', 'T1046', 'T1091', 'T1542.001', 'T1569.002', 'T1137.002', 'T1222', 'T1596.001', 'T1195', 'T1587.002', 'T1491.002', 'T1216.001', 'T1548.001', 'T1003.006', 'T1136', 'T1565.003', 'T1218.002', 'T1555.002', 'T1078.001', 'T1546.001', 'T1600', 'T1557.002', 'T1090.002', 'T1614.001', 'T1558.004', 'T1036.007', 'T1505.002', 'T1010', 'T1564.007', 'T1529', 'T1565', 'T1564.005', 'T1586', 'T1557', 'T1598', 'T1547.008', 'T1601.002', 'T1218.008', 'T1137.001', 'T1597.002', 'T1578.004', 'T1537', 'T1586.002', 'T1547.002', 'T1036.002', 'T1185', 'T1574', 'T1027.002', 'T1052', 'T1135', 'T1588', 'T1098.004', 'T1027.003', 'T1497.001', 'T1586.001', 'T1016', 'T1600.002', 'T1137.005', 'T1008', 'T1136.003', 'T1003.007', 'T1583.005', 'T1048.001', 'T1601', 'T1606', 'T1133', 'T1564.004', 'T1574.008', 'T1612', 'T1037.003', 'T1574.002', 'T1542.002', 'T1542', 'T1048.003', 'T1059.007', 'T1218.011', 'T1583.001', 'T1071.002', 'T1070', 'T1037.001', 'T1083', 'T1071.003', 'T1546.008', 'T1552.005', 'T1587', 'T1095', 'T1589.001', 'T1482', 'T1003.001', 'T1497.003', 'T1557.001', 'T1021.005', 'T1036.004', 'T1602.001', 'T1557.003', 'T1528', 'T1486', 'T1485', 'T1583', 'T1078.003', 'T1055.012', 'T1566', 'T1222.001', 'T1053.006', 'T1036.003', 'T1016.001', 'T1055.003', 'T1221', 'T1055.013', 'T1218.001', 'T1218.014', 'T1190', 'T1553.003', 'T1571', 'T1140', 'T1033', 'T1218.007', 'T1059.001', 'T1591.001', 'T1056', 'T1011', 'T1596', 'T1078.002', 'T1591.004', 'T1547', 'T1561.002', 'T1082', 'T1543.004', 'T1547.010', 'T1090.004', 'T1069.002', 'T1555', 'T1570', 'T1078.004', 'T1608', 'T1021.006', 'T1480', 'T1560.002', 'T1608.004', 'T1547.003', 'T1569', 'T1565.002', 'T1218.005', 'T1110.001', 'T1583.002', 'T1134.001', 'T1539', 'T1550.004', 'T1087', 'T1597', 'T1505.004', 'T1606.002', 'T1069.001', 'T1087.003', 'T1484.001', 'T1505.003', 'T1543.001', 'T1593', 'T1614', 'T1499.004', 'T1568.002', 'T1546.003', 'T1059.005', 'T1580', 'T1553.004', 'T1552', 'T1213.002', 'T1589', 'T1071', 'T1597.001', 'T1554', 'T1569.001', 'T1601.001', 'T1584', 'T1036', 'T1584.002', 'T1572', 'T1556.003', 'T1036.006', 'T1591.002', 'T1199', 'T1547.015', 'T1552.006', 'T1134', 'T1074.002', 'T1216', 'T1620', 'T1057', 'T1055.011', 'T1548.003', 'T1564', 'T1218.009', 'T1563', 'T1590.004', 'T1552.004', 'T1005', 'T1021.002', 'T1564.002', 'T1547.013', 'T1070.001', 'T1613', 'T1588.005', 'T1025', 'T1127.001', 'T1212', 'T1205.001', 'T1543', 'T1562', 'T1014', 'T1562.004', 'T1119', 'T1610', 'T1550.002', 'T1546.002', 'T1111', 'T1560.001', 'T1547.005', 'T1592.004', 'T1059', 'T1498', 'T1037.004', 'T1552.007', 'T1136.002', 'T1113', 'T1587.003', 'T1548', 'T1090.003', 'T1592', 'T1564.006', 'T1556.004', 'T1590.005', 'T1589.003', 'T1562.010', 'T1578', 'T1562.009', 'T1562.006', 'T1598.001', 'T1592.001', 'T1110', 'T1069.003', 'T1546.015', 'T1497.002', 'T1584.006', 'T1137.003', 'T1556.001', 'T1059.004', 'T1556.002', 'T1602.002', 'T1593.002', 'T1583.003', 'T1574.001', 'T1134.005', 'T1518', 'T1197', 'T1036.005', 'T1558', 'T1007', 'T1001.003', 'T1053', 'T1217', 'T1560.003']
TACTICS_TECHNIQUES_RELATIONSHIP_DF = {"TA0001":pd.Series(['T1189','T1190','T1133','T1200','T1566','T1566.001','T1566.002','T1566.003','T1091','T1195','T1195.001','T1195.002','T1195.003','T1199','T1078','T1078.001','T1078.002','T1078.003','T1078.004']),
"TA0002":pd.Series(['T1059','T1059.001','T1059.002','T1059.003','T1059.004','T1059.005','T1059.006','T1059.007','T1059.008','T1609','T1610','T1203','T1559','T1559.001','T1559.002','T1559.003','T1106','T1053','T1053.002','T1053.003','T1053.005','T1053.006','T1053.007','T1129','T1072','T1569','T1569.001','T1569.002','T1204','T1204.001','T1204.002','T1204.003','T1047']),
"TA0003":pd.Series(['T1098','T1098.001','T1098.002','T1098.003','T1098.004','T1098.005','T1197','T1547','T1547.001','T1547.002','T1547.003','T1547.004','T1547.005','T1547.006','T1547.007','T1547.008','T1547.009','T1547.010','T1547.012','T1547.013','T1547.014','T1547.015','T1037','T1037.001','T1037.002','T1037.003','T1037.004','T1037.005','T1176','T1554','T1136','T1136.001','T1136.002','T1136.003','T1543','T1543.001','T1543.002','T1543.003','T1543.004','T1546','T1546.001','T1546.002','T1546.003','T1546.004','T1546.005','T1546.006','T1546.007','T1546.008','T1546.009','T1546.010','T1546.011','T1546.012','T1546.013','T1546.014','T1546.015','T1133','T1574','T1574.001','T1574.002','T1574.004','T1574.005','T1574.006','T1574.007','T1574.008','T1574.009','T1574.010','T1574.011','T1574.012','T1574.013','T1525','T1556','T1556.001','T1556.002','T1556.003','T1556.004','T1556.005','T1137','T1137.001','T1137.002','T1137.003','T1137.004','T1137.005','T1137.006','T1542','T1542.001','T1542.002','T1542.003','T1542.004','T1542.005','T1053','T1053.002','T1053.003','T1053.005','T1053.006','T1053.007','T1505','T1505.001','T1505.002','T1505.003','T1505.004','T1505.005','T1205','T1205.001','T1078','T1078.001','T1078.002','T1078.003','T1078.004']),
"TA0004":pd.Series(['T1548','T1548.001','T1548.002','T1548.003','T1548.004','T1134','T1134.001','T1134.002','T1134.003','T1134.004','T1134.005','T1547','T1547.001','T1547.002','T1547.003','T1547.004','T1547.005','T1547.006','T1547.007','T1547.008','T1547.009','T1547.010','T1547.012','T1547.013','T1547.014','T1547.015','T1037','T1037.001','T1037.002','T1037.003','T1037.004','T1037.005','T1543','T1543.001','T1543.002','T1543.003','T1543.004','T1484','T1484.001','T1484.002','T1611','T1546','T1546.001','T1546.002','T1546.003','T1546.004','T1546.005','T1546.006','T1546.007','T1546.008','T1546.009','T1546.010','T1546.011','T1546.012','T1546.013','T1546.014','T1546.015','T1068','T1574','T1574.001','T1574.002','T1574.004','T1574.005','T1574.006','T1574.007','T1574.008','T1574.009','T1574.010','T1574.011','T1574.012','T1574.013','T1055','T1055.001','T1055.002','T1055.003','T1055.004','T1055.005','T1055.008','T1055.009','T1055.011','T1055.012','T1055.013','T1055.014','T1055.015','T1053','T1053.002','T1053.003','T1053.005','T1053.006','T1053.007','T1078','T1078.001','T1078.002','T1078.003','T1078.004']),
"TA0005":pd.Series(['T1548','T1548.001','T1548.002','T1548.003','T1548.004','T1134','T1134.001','T1134.002','T1134.003','T1134.004','T1134.005','T1197','T1612','T1622','T1140','T1610','T1006','T1484','T1484.001','T1484.002','T1480','T1480.001','T1211','T1222','T1222.001','T1222.002','T1564','T1564.001','T1564.002','T1564.003','T1564.004','T1564.005','T1564.006','T1564.007','T1564.008','T1564.009','T1564.010','T1574','T1574.001','T1574.002','T1574.004','T1574.005','T1574.006','T1574.007','T1574.008','T1574.009','T1574.010','T1574.011','T1574.012','T1574.013','T1562','T1562.001','T1562.002','T1562.003','T1562.004','T1562.006','T1562.007','T1562.008','T1562.009','T1562.010','T1070','T1070.001','T1070.002','T1070.003','T1070.004','T1070.005','T1070.006','T1202','T1036','T1036.001','T1036.002','T1036.003','T1036.004','T1036.005','T1036.006','T1036.007','T1556','T1556.001','T1556.002','T1556.003','T1556.004','T1556.005','T1578','T1578.001','T1578.002','T1578.003','T1578.004','T1112','T1601','T1601.001','T1601.002','T1599','T1599.001','T1027','T1027.001','T1027.002','T1027.003','T1027.004','T1027.005','T1027.006','T1647','T1542','T1542.001','T1542.002','T1542.003','T1542.004','T1542.005','T1055','T1055.001','T1055.002','T1055.003','T1055.004','T1055.005','T1055.008','T1055.009','T1055.011','T1055.012','T1055.013','T1055.014','T1055.015','T1620','T1207','T1014','T1553','T1553.001','T1553.002','T1553.003','T1553.004','T1553.005','T1553.006','T1218','T1218.001','T1218.002','T1218.003','T1218.004','T1218.005','T1218.007','T1218.008','T1218.009','T1218.010','T1218.011','T1218.012','T1218.013','T1218.014','T1216','T1216.001','T1221','T1205','T1205.001','T1127','T1127.001','T1535','T1550','T1550.001','T1550.002','T1550.003','T1550.004','T1078','T1078.001','T1078.002','T1078.003','T1078.004','T1497','T1497.001','T1497.002','T1497.003','T1600','T1600.001','T1600.002','T1220']),
"TA0006":pd.Series(['T1557','T1557.001','T1557.002','T1557.003','T1110','T1110.001','T1110.002','T1110.003','T1110.004','T1555','T1555.001','T1555.002','T1555.003','T1555.004','T1555.005','T1212','T1187','T1606','T1606.001','T1606.002','T1056','T1056.001','T1056.002','T1056.003','T1056.004','T1556','T1556.001','T1556.002','T1556.003','T1556.004','T1556.005','T1111','T1621','T1040','T1003','T1003.001','T1003.002','T1003.003','T1003.004','T1003.005','T1003.006','T1003.007','T1003.008','T1528','T1558','T1558.001','T1558.002','T1558.003','T1558.004','T1539','T1552','T1552.001','T1552.002','T1552.003','T1552.004','T1552.005','T1552.006','T1552.007']),
"TA0007":pd.Series(['T1087','T1087.001','T1087.002','T1087.003','T1087.004','T1010','T1217','T1580','T1538','T1526','T1619','T1613','T1622','T1482','T1083','T1615','T1046','T1135','T1040','T1201','T1120','T1069','T1069.001','T1069.002','T1069.003','T1057','T1012','T1018','T1518','T1518.001','T1082','T1614','T1614.001','T1016','T1016.001','T1049','T1033','T1007','T1124','T1497','T1497.001','T1497.002','T1497.003']),
"TA0008":pd.Series(['T1210','T1534','T1570','T1563','T1563.001','T1563.002','T1021','T1021.001','T1021.002','T1021.003','T1021.004','T1021.005','T1021.006','T1091','T1072','T1080','T1550','T1550.001','T1550.002','T1550.003','T1550.004']),
"TA0009":pd.Series(['T1557','T1557.001','T1557.002','T1557.003','T1560','T1560.001','T1560.002','T1560.003','T1123','T1119','T1185','T1115','T1530','T1602','T1602.001','T1602.002','T1213','T1213.001','T1213.002','T1213.003','T1005','T1039','T1025','T1074','T1074.001','T1074.002','T1114','T1114.001','T1114.002','T1114.003','T1056','T1056.001','T1056.002','T1056.003','T1056.004','T1113','T1125']),
"TA0010":pd.Series(['T1020','T1020.001','T1030','T1048','T1048.001','T1048.002','T1048.003','T1041','T1011','T1011.001','T1052','T1052.001','T1567','T1567.001','T1567.002','T1029','T1537']),
"TA0011":pd.Series(['T1071','T1071.001','T1071.002','T1071.003','T1071.004','T1092','T1132','T1132.001','T1132.002','T1001','T1001.001','T1001.002','T1001.003','T1568','T1568.001','T1568.002','T1568.003','T1573','T1573.001','T1573.002','T1008','T1105','T1104','T1095','T1571','T1572','T1090','T1090.001','T1090.002','T1090.003','T1090.004','T1219','T1205','T1205.001','T1102','T1102.001','T1102.002','T1102.003']),
"TA0040":pd.Series(['T1531','T1485','T1486','T1565','T1565.001','T1565.002','T1565.003','T1491','T1491.001','T1491.002','T1561','T1561.001','T1561.002','T1499','T1499.001','T1499.002','T1499.003','T1499.004','T1495','T1490','T1498','T1498.001','T1498.002','T1496','T1489','T1529']),                
"TA0043":pd.Series(['T1595','T1595.001','T1595.002','T1595.003','T1592','T1592.001','T1592.002','T1592.003','T1592.004','T1589','T1589.001','T1589.002','T1589.003','T1590','T1590.001','T1590.002','T1590.003','T1590.004','T1590.005','T1590.006','T1591','T1591.001','T1591.002','T1591.003','T1591.004','T1598','T1598.001','T1598.002','T1598.003','T1597','T1597.001','T1597.002','T1596','T1596.001','T1596.002','T1596.003','T1596.004','T1596.005','T1593','T1593.001','T1593.002','T1594']),
"TA0042":pd.Series(['T1583','T1583.001','T1583.002','T1583.003','T1583.004','T1583.005','T1583.006','T1586','T1586.001','T1586.002','T1584','T1584.001','T1584.002','T1584.003','T1584.004','T1584.005','T1584.006','T1587','T1587.001','T1587.002','T1587.003','T1587.004','T1585','T1585.001','T1585.002','T1588','T1588.001','T1588.002','T1588.003','T1588.004','T1588.005','T1588.006','T1608','T1608.001','T1608.002','T1608.003','T1608.004','T1608.005'])
}

df_tram = pd.read_csv('tram__with_all_labels.csv', encoding='utf-8')
df_attack = pd.read_csv('attack_with_all_labels.csv', encoding='utf-8')

df_tram['tactic_label'] = df_tram.apply(lambda x: list(x[TACTIC]), axis=1)
df_tram['technique_label'] = df_tram.apply(lambda x: list(x[TECHNIQUE]), axis=1)
df_attack['tactic_label'] = df_attack.apply(lambda x: list(x[TACTIC]), axis=1)
df_attack['technique_label'] = df_attack.apply(lambda x: list(x[TECHNIQUE]), axis=1)

df = pd.concat([df_tram, df_attack], ignore_index=True)


from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

def ioc_sub(text):
    def reg_handler(obj):
        s = obj.group(1)
        s = ' '.join(s.split('\\'))
        return s

    def file_handler(obj):
        s = obj.group(2)
        s = s.split('\\')[-1]
        return s
    
    text = re.sub(r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|\[\.\])){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
    text = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
    text = re.sub(r'\b(CVE\-[0-9]{4}\-[0-9]{4,6})\b', 'CVE', text)
    text = re.sub(r'CVE-[0-9]{4}-[0-9]{4,6}', 'vulnerability', text)
    text = re.sub(r'\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
    text = re.sub(r'\b((HKLM|HKCU|HKCR|HKU|HKCC)\\[\\A-Za-z0-9-_]+)\b', reg_handler, text)
    text = re.sub(r'\b([a-zA-Z]{1}:\\([0-9a-zA-Z_\.\-\/\\]+))\b', file_handler, text)
    text = re.sub(r'\b([a-f0-9]{32}|[A-F0-9]{32})\b', 'MD5', text)
    text = re.sub(r'\b([a-f0-9]{40}|[A-F0-9]{40})\b', 'SHA1', text)
    text = re.sub(r'\b([a-f0-9]{64}|[A-F0-9]{64})\b', 'SHA256', text)
    text = re.sub(r'\d+:[A-Za-z0-9/+]+:[A-Za-z0-9/+]+', 'ssdeep', text)
    text = re.sub(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
    text = re.sub(r'h[tx][tx]ps?:[\\\/][\\\/](?:[0-9a-zA-Z_\.\-\/\\]|\[\.\])+', 'URL', text)
    text = re.sub(r'((?:[0-9a-zA-Z_\-]+\.)+(?:(?!exe|dll)[a-z]{2,4}))', 'domain', text)
    text = re.sub(r'[a-fA-F0-9]{16}', '', text)
    text = re.sub(r'[0-9]{8}', '', text)
    text = re.sub(r'x[A-Fa-f0-9]{2}', '', text)
    
    return text

def rmstopword_and_lemmatize(text):
    token = [word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')]
    tag = pos_tag(token)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x[0], pos=get_wordnet_pos(x[1])), tag)))
    
    return text

def preprocess(text):
    text = str(text)
    #text = text.lower()
    text = re.sub("\r\n", " ", text)
    text = re.sub('etc\.', '', text)
    text = re.sub('et al\.', '', text)
    text = re.sub('e\.g\.', '', text)
    text = re.sub('i\.e\.', '', text)
    #text = re.sub(r'\[.\]', '.', text)
    text = re.sub(r'\[\d+\]', '', text)
    
    text = ioc_sub(text)
    
    text = re.sub(r'[^A-Za-z0-9_\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = rmstopword_and_lemmatize(text)
    
    text = re.sub(r'[\[\]]', ' ', text)
    text = re.sub(r' [a-z0-9] ', '', text)
    
    return text


df['text_clean'] = df['text'].map(lambda t: preprocess(t))


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=2222)
validation, test = train_test_split(test, test_size=0.5, random_state=2222)

df_te_train = df.iloc[train.index][['text_clean', 'technique_label']]
df_te_val = df.iloc[validation.index][['text_clean', 'technique_label']]
df_te_test = df.iloc[test.index][['text_clean', 'technique_label']]
df_te_train = df_te_train.rename(columns={'text_clean':'text', 'technique_label':'labels'})
df_te_val = df_te_val.rename(columns={'text_clean':'text', 'technique_label':'labels'})
df_te_test = df_te_test.rename(columns={'text_clean':'text', 'technique_label':'labels'})


# Technique_multi model
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
te_multi_model_args = MultiLabelClassificationArgs()
te_multi_model_args.reprocess_input_data = True
te_multi_model_args.overwrite_output_dir = True
te_multi_model_args.evaluate_during_training = True
te_multi_model_args.use_multiprocessing = False
te_multi_model_args.use_multiprocessing_for_evaluation = False
te_multi_model_args.use_multiprocessed_decoding = False
te_multi_model_args.train_batch_size = 16
te_multi_model_args.eval_batch_size = 16
te_multi_model_args.num_train_epochs = 32
te_multi_model_args.use_early_stopping = True
te_multi_model_args.early_stopping_delta = 0.01
te_multi_model_args.early_stopping_metric = "eval_loss"
te_multi_model_args.early_stopping_metric_minimize = True
te_multi_model_args.early_stopping_patience = 10
te_multi_model_args.evaluate_during_training_steps = 1000
te_multi_model_args.learning_rate = 3e-5 
te_multi_model_args.output_dir = './outputs/technique_multi/'

from sklearn.metrics import coverage_error, label_ranking_loss
te_multi_model = MultiLabelClassificationModel(
    'distilbert',
    'distilbert/finetune',
    use_cuda=True,
    num_labels=len(TECHNIQUE),
    args=te_multi_model_args,
)

te_multi_model.train_model(df_te_train, eval_df=df_te_test, coverr=coverage_error, lrloss=label_ranking_loss)
te_result, te_model_outputs, te_wrong_predictions = te_multi_model.eval_model(df_te_test)

from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import accuracy_score

te_multi_model = MultiLabelClassificationModel(
    'distilbert',
    './outputs/technique_multi/',
)
te_predictions, te_model_outputs = te_multi_model.predict(list(df_te_test['text']))

te_true = np.array([])
te_true = np.append(te_true, [row for row in df_te_test['labels']])
te_true = te_true.reshape(len(te_model_outputs), -1)

print('Technique------------------------------------------------------')
print('Coverage error: %f' % coverage_error(te_true, te_model_outputs))
print('LRAP: %f' % label_ranking_average_precision_score(te_true, te_model_outputs))
print('Label ranking loss: %f' % label_ranking_loss(te_true, te_model_outputs))

te_model_outputs_bi = te_predictions
print('Hamming loss: %f' % hamming_loss(te_true, te_model_outputs_bi))
print('Precision score(samples): %f' % precision_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('Precision score(macro): %f' % precision_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('Precision score(micro): %f' % precision_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('Recall score(samples): %f' % recall_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('Recall score(macro): %f' % recall_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('Recall score(micro): %f' % recall_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('F1 score(samples): %f' % f1_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('F1 score(macro): %f' % f1_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('F1 score(micro): %f' % f1_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('F0.5 score(samples): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='samples', zero_division=0))
print('F0.5 score(macro): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='macro', zero_division=0))
print('F0.5 score(micro): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='micro', zero_division=0))
print('Accuracy score: %f' % accuracy_score(te_true, te_model_outputs_bi))


df_ta_train = train[['text_clean', 'tactic_label']]
df_ta_val = validation[['text_clean', 'tactic_label']]
df_ta_test = test[['text_clean', 'tactic_label']]
df_ta_train = df_ta_train.rename(columns={'text_clean':'text', 'tactic_label':'labels'})
df_ta_val = df_ta_val.rename(columns={'text_clean':'text', 'tactic_label':'labels'})
df_ta_test = df_ta_test.rename(columns={'text_clean':'text', 'tactic_label':'labels'})

# Tactic_multi model
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
ta_multi_model_args = MultiLabelClassificationArgs()
ta_multi_model_args.reprocess_input_data = True
ta_multi_model_args.overwrite_output_dir = True
ta_multi_model_args.evaluate_during_training = True
ta_multi_model_args.manual_seed = 4
ta_multi_model_args.use_multiprocessing = False
ta_multi_model_args.use_multiprocessing_for_evaluation = False
ta_multi_model_args.use_multiprocessed_decoding = False
ta_multi_model_args.train_batch_size = 16
ta_multi_model_args.eval_batch_size = 16
ta_multi_model_args.num_train_epochs = 32
ta_multi_model_args.learning_rate = 5e-5
ta_multi_model_args.use_early_stopping = True
ta_multi_model_args.early_stopping_delta = 0.01
ta_multi_model_args.early_stopping_metric = "eval_loss"
ta_multi_model_args.early_stopping_metric_minimize = True
ta_multi_model_args.early_stopping_patience = 6
ta_multi_model_args.evaluate_during_training_steps = 1000
ta_multi_model_args.output_dir = './outputs/tactic_multi/'

from sklearn.metrics import coverage_error, label_ranking_loss
ta_multi_model = MultiLabelClassificationModel(
    'distilbert',
    'distilbert/finetune/',
    use_cuda=True,
    num_labels=len(TACTIC),
    args=ta_multi_model_args,
)

ta_multi_model.train_model(df_ta_train, eval_df=df_ta_test, coverr=coverage_error, lrloss=label_ranking_loss)
ta_result, ta_model_outputs, ta_wrong_predictions = ta_multi_model.eval_model(df_ta_test)

from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

ta_multi_model = MultiLabelClassificationModel(
    'distilbert',
    './outputs/tactic_multi'
)
ta_model_outputs = ta_multi_model.predict(list(df_ta_test['text']))[1]

ta_true = np.array([])
ta_true = np.append(ta_true, [row for row in df.iloc[df_ta_test.index]['tactic_label']])
ta_true = ta_true.reshape(len(ta_model_outputs), -1)

print('Tactic--------------------------------------------------------')
print('Coverage error: %f' % coverage_error(ta_true, ta_model_outputs))
print('LRAP: %f' % label_ranking_average_precision_score(ta_true, ta_model_outputs))
print('Label ranking loss: %f' % label_ranking_loss(ta_true, ta_model_outputs))

ta_model_outputs_bi = (ta_model_outputs > 0.5).astype(np.int_)
print('Hamming loss: %f' % hamming_loss(ta_true, ta_model_outputs_bi))
print('Precision score(samples): %f' % precision_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('Precision score(macro): %f' % precision_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('Precision score(micro): %f' % precision_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('Recall score(samples): %f' % recall_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('Recall score(macro): %f' % recall_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('Recall score(micro): %f' % recall_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('F1 score(samples): %f' % f1_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('F1 score(macro): %f' % f1_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('F1 score(micro): %f' % f1_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('F0.5 score(samples): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='samples', zero_division=0))
print('F0.5 score(macro): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='macro', zero_division=0))
print('F0.5 score(micro): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='micro', zero_division=0))
print('Accuracy score: %f' % accuracy_score(ta_true, ta_model_outputs_bi))


# Post-processing
print('Post-processing-----------------------------------------------------')
ta_correct_true = {}
ta_correct_false = {}
sub_correct_true = {}
sub_correct_false = {}
highrate_correct_true = {}
highrate_correct_false = {}
all_true_mod = 0
all_false_mod = 0
all_true = 0
all_false = 0
true_origin = 0
false_origin = 0

te_modified = []
modified_ind = set()
for ind in range(len(df_te_test)):
    te_tmp = {}
    te_mask = {}
    te_pred = {}
    te_real = {}
    ta_real = {}
    ta_pred = {}

    ta_threshold = 0.01
    te_threshold = 0.25
    
    for i, v in enumerate(ta_model_outputs[ind]):
        ta_pred[TACTIC[i]] = v
        for te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[TACTIC[i]]:
            try:
                te_mask[te] |= int(v>ta_threshold)
            except KeyError:
                te_mask[te] = int(v>ta_threshold)
    for te, v in list(zip(TECHNIQUE, te_model_outputs[ind])):
        te_pred[te] = v
    for te, v in list(zip(TECHNIQUE, te_true[ind])):
        te_real[te] = int(v)
    for ta, v in list(zip(TACTIC, ta_true[ind])):
        ta_real[ta] = int(v)

    tp = fp = tn = fn = 0
    tp_ = fp_ = tn_ = fn_ = 0
    for te in TECHNIQUE:
        # if te was set to True by sub-tech then pass
        try:
            if te_tmp[te]:
                continue
        except KeyError:
            # Te prediction>0.95 then discard correction
            if te_pred[te] > 0.95:
                te_mask[te] = 1
                if te_real[te]:
                    try:
                        highrate_correct_true[te] += 1
                    except KeyError:
                        highrate_correct_true[te] = 1
                    # print(f'{ind}: {te}, te_real={te_real[te]} te_pred={te_pred[te]} high pred rate -> True')
                else:
                    try:
                        highrate_correct_false[te] += 1
                    except KeyError:
                        highrate_correct_false[te] = 1
                    # print(f'{ind}: {te}, te_real={te_real[te]} te_pred={te_pred[te]} high pred rate -> False')
                    
            te_tmp[te] = int(te_pred[te]>=te_threshold) & te_mask[te]
        
        # Te set to True if te_pred>=threshold
        if te_pred[te] >= te_threshold:
            if te_real[te]:
                tp += 1 # real=1 and pred=1 => TP
                if te_mask[te]:
                    tp_ += 1 # real=1 and pred=1->1 => TP_
                    
                    # Set parent technique to True if sub-tech is TP_
                    if len(te.split('.')) > 1:
                        te_parent = te.split('.')[0]
                        try:
                            if te_tmp[te_parent]:
                                pass
                            else:
                                raise KeyError
                        except KeyError:
                            te_tmp[te_parent] = 1

                            if te_real[te_parent]:
                                if te_pred[te_parent] < te_threshold:
                                    try:
                                        sub_correct_true[te_parent] += 1
                                    except KeyError:
                                        sub_correct_true[te_parent] = 1
                                    status = 'True'
                                else:
                                    status = 'Useless'
                            else:
                                try:
                                    sub_correct_false[te_parent] += 1
                                except KeyError:
                                    sub_correct_false[te_parent] = 1
                                status = 'False'
                            if status != 'Useless':
                                print(f'{ind}: {te_parent}: {te}, te_parent_real={te_real[te_parent]} te_parent_pred={te_pred[te_parent]}, te_real={te_real[te]} te_pred={te_pred[te]} -> {status}')
                        
                else:
                    fn_ += 1 # real=1 and pred=1->0 => FN_
                    
                    for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                        if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                            print(f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> false')
                            break
                    try:
                        ta_correct_false[te] += 1
                    except KeyError:
                        ta_correct_false[te] = 1
            else:
                fp += 1 # real=0 and pred=1 => FP
                if te_mask[te]:
                    fp_ += 1 # real=0 and pred=1->1 => FP_
                else:
                    tn_ += 1 # real=0 and pred=1->0 => TN_
                    
                    for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                        if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                            print(f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> true')
                            break
                    try:
                        ta_correct_true[te] += 1
                    except KeyError:
                        ta_correct_true[te] = 1
                        
        # Te set to False if te_pred<threshold
        else:
            if te_real[te]:
                fn += 1 # real=1 and pred=0 => FN
                fn_ += 1 # real=1 and pred=0->0 => FN_
            else:
                tn += 1 # real=0 and pred=0 => TN
                tn_ += 1 # real=0 and pred=0->0 =>TN_
    
    true_mod = 0
    false_mod = 0
    true = 0
    false = 0
    true_ori = 0
    false_ori = 0
    for te in TECHNIQUE:
        if te_real[te]:
            if te_pred[te]>=0.5:
                true_ori += 1
                if te_tmp[te]:
                    true += 1
                else:
                    false += 1
                    false_mod += 1
            else:
                false_ori += 1
                if te_tmp[te]:
                    true += 1
                    true_mod += 1
                else:
                    false += 1
        # else:
        #     if te_pred[te]>=0.5:
        #         false_ori += 1
        #         if te_tmp[te]:
        #             false += 1
        #         else:
        #             true += 1
        #             true_mod += 1
        #     else:
        #         true_ori += 1
        #         if te_tmp[te]:
        #             false += 1
        #             false_mod += 1
        #         else:
        #             true += 1
            
    all_true_mod += true_mod
    all_false_mod += false_mod
    all_true += true
    all_false += false
    true_origin += true_ori
    false_origin += false_ori
    # print(f'{ind}: true modified {true}, false modified {false}')

    te_tmp = [te_tmp[te] for te in TECHNIQUE]
    te_modified.append(te_tmp)
    # print([[tp, fn],     #        [fp, tn]], '\n', 
    #      [[tp_, fn_],     #       [fp_, tn_]])

te_model_outputs_bi = (te_model_outputs > 0.5).astype(np.int_)
print('Hamming loss: %f -> %f' % (hamming_loss(te_true, te_model_outputs_bi), hamming_loss(te_true, te_modified)))
print('Precision score(samples): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), precision_score(te_true, te_modified, average='samples', zero_division=0)))
print('Precision score(macro): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), precision_score(te_true, te_modified, average='macro', zero_division=0)))
print('Precision score(micro): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), precision_score(te_true, te_modified, average='micro', zero_division=0)))

print('Recall score(samples): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), recall_score(te_true, te_modified, average='samples', zero_division=0)))
print('Recall score(macro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), recall_score(te_true, te_modified, average='macro', zero_division=0)))
print('Recall score(micro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), recall_score(te_true, te_modified, average='micro', zero_division=0)))

print('F1 score(samples): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), f1_score(te_true, te_modified, average='samples', zero_division=0)))
print('F1 score(macro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), f1_score(te_true, te_modified, average='macro', zero_division=0)))
print('F1 score(micro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), f1_score(te_true, te_modified, average='micro', zero_division=0)))

print('F0.5 score(samples): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='samples', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='samples', zero_division=0)))
print('F0.5 score(macro): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='macro', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='macro', zero_division=0)))
print('F0.5 score(micro): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='micro', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='micro', zero_division=0)))

print('Accuracy score: %f -> %f' % (accuracy_score(te_true, te_model_outputs_bi), accuracy_score(te_true, te_modified)))