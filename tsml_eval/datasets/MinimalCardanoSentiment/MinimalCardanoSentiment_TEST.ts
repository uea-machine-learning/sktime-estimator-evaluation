% This is a cut down version of the problem CardanoSentiment, useful for code examples and unit tests
% The train set is reduced from 74 cases to 20 cases and the test set is reduced from 33 to 20
%
@problemName CardanoSentiment
@timestamps false
@univariate false
@equalLength true
@seriesLength 24
@targetlabel true
@data
0.52654,0.527,0.5271,0.5271,0.53274,0.539,0.54262,0.54065,0.5455,0.54216,0.54395,0.54662,0.54089,0.54309,0.53953,0.53553,0.53128,0.53307,0.53288,0.53288,0.53886,0.53521,0.53521,0.53521:50000.0,56.1789234,439.91244494,0.0,1435.64188905,12654.18715291,12994.01944456,10000.0,22188.72215054,3983.37108399,3217.2572733,9501.16501917,2882.8742118,14357.64233668,4797.94743502,1051.35592684,12989.1481268,100934.80858048,101946.4179015,0.0,5187.45260547,260065.944223,0.0,0.0:0.0795
0.93072,0.93311,0.92462,0.92579,0.93518,0.95121,0.94497,0.94924,0.95412,0.94959,0.94981,0.96047,0.98059,0.96508,0.95118,0.95775,0.95341,0.95341,0.94005,0.93115,0.93457,0.93884,0.94768,0.955:37944.65804979,1655.90980388,9754.60036876,20019.0,1681.17407842,1054.94926912,23244.894043,10309.61020307,30239.0,10351.68977015,1653.18466906,5840.8006979,72206.36003188,7375.4498625,38072.68273367,40048.60631682,3483.21473753,0.0,3899.54458615,125313.43675509,1234.87540023,230.69707195,2364.957588,183.169075:0.2185
0.4698,0.46659,0.46659,0.46929,0.47217,0.4725,0.46948,0.4695,0.46838,0.47178,0.46839,0.465,0.45821,0.45609,0.45672,0.45203,0.45597,0.45698,0.45387,0.45354,0.45675,0.45664,0.46417,0.46597:7987.80015374,50000.0,0.0,887.58852059,20194.70793947,996.44310608,15496.85516779,35.0,426.17555656,30.05589625,1085.78720212,3834.0,321950.54871209,310683.1334605,13632.38672577,2000.0,1272.5538108,9104.12108068,663.48382929,911.25501513,11170.92541895,30039.89872094,12485.83055545,5378.53252006:0.5
1.16889,1.16889,1.1823,1.18061,1.18239,1.18239,1.2161,1.21787,1.22381,1.20141,1.19872,1.21824,1.22461,1.21813,1.23089,1.206,1.20654,1.16766,1.17437,1.1904,1.19659,1.20084,1.19797,1.21195:3884.06984379,0.0,282.77925192,308.34149855,8.81486673,0.0,7182.51853648,48279.27281877,1891.41819593,15244.59927839,4743.17837322,4189.28573019,14139.96889992,3448.85452066,4977.27851515,50825.38829507,2584.50862489,9368.20055544,2534.74500749,31403.66957334,2807.96807405,65.48028,7281.39558633,7840.28855858:0.3198
0.82009,0.82009,0.83737,0.83737,0.83315,0.83711,0.84185,0.85237,0.84275,0.85107,0.84421,0.83737,0.83452,0.8501,0.83339,0.84096,0.84102,0.8369,0.8321,0.83157,0.83894,0.84288,0.83881,0.84085:18038.32476443,0.0,28431.552943,0.0,6889.0,163.877108,8128.22161722,27539.44521306,16499.26255896,2493.64654065,26582.260893,34014.72680524,677.13008667,21049.16905734,74368.72281617,9826.46102615,5395.29762243,6004.82026739,76490.08185914,42656.80194182,15054.17925706,105.21500481,3279.93484621,1411.84539454:0.3538
0.4599,0.4599,0.4599,0.4599,0.4599,0.4599,0.46516,0.46682,0.46682,0.46511,0.46341,0.46341,0.46329,0.46329,0.4665,0.46133,0.46133,0.46142,0.46513,0.4662,0.47079,0.46922,0.46542,0.46542:130000.0,0.0,0.0,0.0,0.0,0.0,100094.68117035,1674.52423923,0.0,50031.88493044,2933.48351146,2701.52620191,210.15184873,0.0,106.6984149,2727.81054822,0.0,820.10047681,14036.43846735,321.46226553,2767.20963429,795.98241215,11057.04079763,0.0:0.931
0.44252,0.44782,0.44726,0.44769,0.44495,0.45036,0.45036,0.44635,0.4466,0.44379,0.44585,0.4497,0.46165,0.45313,0.44325,0.4476,0.44606,0.44971,0.44971,0.45436,0.4563,0.45779,0.45602,0.45665:16244.40496377,670.40364008,8858.24910091,955.28681298,51.10921744,14138.05666974,0.0,368.62598467,50550.0,219.97558145,874.91236963,8221.07494653,12105.92509473,18464.03975643,29364.06962404,124.77738718,50507.37222007,2455.54771974,0.0,50000.0,19997.05304148,276893.73601054,5904.63797247,2824.96424543:0.5305
1.03131,1.0345,1.03293,1.03173,1.03138,1.03534,1.03618,1.03265,1.0375,1.0375,1.03283,1.03788,1.04166,1.03867,1.0385,1.02507,1.03216,1.03562,1.03387,1.03128,1.03206,1.03202,1.03628,1.04237:63571.68211827,4343.12303152,6138.55277999,1500.37164955,5700.59723779,524.02652268,77.67025999,557.43313155,4529.02842471,0.0,3965.95770391,300099.37475248,19928.44367032,32.505719,2812.06727834,804.22939441,5669.62221491,3197.50758864,564.6916274,2178.47405374,229.00497084,2731.95884004,2697.0,5469.27784551:0.3833
0.94402,0.94116,0.94116,0.94116,0.94116,0.94116,0.93008,0.92756,0.92756,0.92756,0.92756,0.9328,0.93505,0.94592,0.94919,0.94955,0.94902,0.94,0.9408,0.94654,0.94611,0.94611,0.9521,0.9521:19358.91111433,2834.9253483,0.0,0.0,0.0,0.0,710.32596,3245.89626422,0.0,0.0,0.0,56.0,35.609132,11145.03507716,11905.80090054,32739.41232973,11580.30558545,53185.43199228,25000.0,500.0,1057.17100362,0.0,1050.12829954,0.0:0.2076
0.50373,0.50989,0.50777,0.50777,0.50982,0.50872,0.50872,0.51228,0.51319,0.51056,0.51,0.51112,0.53153,0.53528,0.53802,0.53732,0.53347,0.53431,0.53,0.53,0.53289,0.53568,0.5372,0.53532:2981.91408659,1199.31174495,85.468868,0.0,2453.0,168.392304,0.0,15569.71659018,263127.5620347,29503.10751965,1136.34944442,37.4584555,8526.10772717,11683.85306141,1108.36036445,3240.30733541,250350.59469543,5475.91482472,146126.51223766,0.0,666.24518738,2382.5548754,24.56872673,226.224793:0.0
0.43506,0.43541,0.43258,0.43449,0.43282,0.4308,0.4288,0.4288,0.42201,0.41876,0.42171,0.41946,0.42059,0.41682,0.42222,0.43256,0.43017,0.4373,0.44206,0.43961,0.44086,0.43697,0.43697,0.44173:11585.55241608,4428.5003166,1085.46000803,2592.39384088,3322.74480064,7274.31860626,16171.28052238,0.0,19405.16104735,13672.99352366,4469.01212,1015.99252514,1493.96458214,683.1783769,14329.3715252,2961.77388754,1531.92318768,9837.77370399,32457.12077434,22227.01808424,9915.85616772,8600.30002878,0.0,85.0:0.1133
0.85526,0.86201,0.86373,0.87348,0.87054,0.86421,0.84398,0.83586,0.83397,0.84,0.84245,0.85662,0.86354,0.85,0.86121,0.8588,0.87511,0.87432,0.88282,0.88682,0.89057,0.89057,0.90275,0.90275:24044.06840286,3930.23613186,6644.30912188,7542.18376283,5723.29523141,8087.14680201,408778.09123259,36900.86584315,2416.41150837,12702.38888348,14966.14375922,31565.668048,3899.92869006,1500.0,4737.5214997,9400.41494469,40796.77123743,59.775751,440.4025668,4413.73326359,14604.4789009,0.0,7284.3770149,0.0:-0.0967
0.58186,0.578,0.58188,0.58948,0.58824,0.58611,0.58611,0.58462,0.59044,0.58582,0.58778,0.58778,0.58778,0.60213,0.61836,0.61784,0.60842,0.61257,0.62172,0.62751,0.63026,0.62225,0.61,0.61618:7690.77921104,3866.45582227,5313.79909079,33968.00445605,5005.12741713,9243.21488691,0.0,465.08021312,1778.62749189,1825.24690092,10013.69196584,0.0,0.0,13037.33716712,3482.12828283,454.52745487,163.54243778,3268.72079274,39289.34549945,48448.46763006,102951.47579795,20285.92646932,9990.51261075,274.07410457:0.199
0.45972,0.45709,0.45876,0.4623,0.46395,0.46395,0.44734,0.45064,0.44571,0.44431,0.43826,0.43958,0.44451,0.43532,0.43923,0.44224,0.45039,0.4483,0.44509,0.44259,0.44045,0.44396,0.44237,0.4597:408.41320449,6743.73306872,1600.33618935,250.12032927,248.88010773,0.0,21333.51525999,95275.12614134,53201.53590666,52692.91214935,30227.68361544,10237.69642006,202800.03177789,16240.86673524,11308.89105379,18561.0,9856.26990034,10619.07398727,4841.67171619,4880.7687567,1421.89093494,3957.16933592,24975.06291687,12752.52099643:0.31
0.4644,0.4644,0.46337,0.46337,0.46337,0.46843,0.467,0.47054,0.47054,0.4668,0.4668,0.46673,0.46836,0.468,0.472,0.47172,0.47172,0.47172,0.49301,0.50252,0.50689,0.50362,0.50823,0.50891:2721.00992439,0.0,2697.63406701,0.0,0.0,2791.09851888,882.26970021,9559.77224653,0.0,7083.93727408,0.0,8114.375512,1555.02167417,32.88461538,51480.94261644,3711.0,0.0,0.0,33258.48388327,39993.70118738,47782.00349278,5033.63606612,2152.17641261,6298.40170199:-0.0293
0.48,0.48807,0.47484,0.46345,0.45796,0.4595,0.4633,0.45617,0.45895,0.45513,0.46939,0.48067,0.47208,0.48156,0.4708,0.47404,0.47134,0.4892,0.50221,0.5119,0.51089,0.52652,0.52503,0.52618:22800.5927211,2160.63843301,3682.67709493,13633.58486126,18557.66544148,164114.61960556,974.76902654,16352.15450197,217453.85128306,457.02106826,72184.42011105,30178.87539079,17385.90549428,6402.89202214,13795.95532477,139356.943135,395038.45878362,191991.69433133,143392.1981705,72398.87942033,17253.21718949,51588.65173871,46803.76188176,3214.10155311:0.2633
0.93119,0.93765,0.93809,0.94647,0.95245,0.95512,0.94738,0.94769,0.94882,0.94882,0.94907,0.94764,0.94764,0.95,0.94388,0.95556,0.95246,0.95574,0.95439,0.95439,0.95284,0.95343,0.95175,0.95175:6337.61002827,941.66946432,3196.62488004,3189.7416667,1627.9168843,1401.08805924,850.0,109.00068112,2312.75720736,0.0,300.0,486.11820135,0.0,1998.0,40242.82061957,27351.49400473,33459.23261222,22320.76954774,115.02681293,0.0,39.101154,685.94733404,1000.0,0.0:0.572
0.51667,0.51656,0.51382,0.51382,0.51382,0.51622,0.51622,0.51622,0.51513,0.53103,0.52395,0.52758,0.52758,0.524,0.52279,0.52646,0.52143,0.51873,0.51749,0.51821,0.51855,0.51855,0.51521,0.51506:10243.7556559,6112.92579984,577.43264473,0.0,0.0,1165.6418218,0.0,0.0,1058.20027758,533.35555945,1904.26293409,76867.04166045,0.0,30.0,5952.14850256,416.68048285,36705.05860259,14699.0891471,17002.16762212,10497.36621009,33.08131332,0.0,4220.43380105,8027.05355409:0.513
0.53528,0.53069,0.52873,0.52873,0.52873,0.53644,0.53253,0.53481,0.53035,0.5227,0.52036,0.51834,0.52111,0.51501,0.51199,0.51226,0.51155,0.51012,0.51151,0.51321,0.51383,0.51465,0.51386,0.5133:94.67322303,2110.90536535,829.79722688,0.0,0.0,1854.91917804,30.0,227141.44158885,1250.84657626,124700.59493925,172834.16814,19309.82510032,72.55848093,12805.87031356,55789.79835156,988.63681696,1377.84099153,4646.14446557,543.62505131,19745.82315685,4490.83591833,686.05762604,10590.18482268,466.05318623:0.2943
0.48957,0.48957,0.48631,0.49379,0.48578,0.49863,0.49886,0.49487,0.50123,0.50024,0.5047,0.4977,0.49796,0.51007,0.50629,0.49976,0.5018,0.49267,0.48898,0.48048,0.48114,0.48149,0.47716,0.48031:2457.23443546,0.0,22302.47823632,7325.0,10578.4652214,18531.88908795,12832.3108164,119656.11325182,20589.79245807,9462.03984171,18625.67840348,38301.3088867,11461.50556392,67056.94008325,95961.88581996,12311.24478276,1091.64348765,6846.88634221,7901.76799277,22544.63737735,15997.84143636,267.5863967,36023.83753973,399.9807929:0.113
0.45084,0.45032,0.45032,0.45032,0.45032,0.45032,0.44739,0.45012,0.446,0.446,0.44542,0.44542,0.44639,0.44639,0.44639,0.44639,0.44639,0.4461,0.4461,0.44238,0.44309,0.441,0.43873,0.42912:0.0,1000.0,0.0,0.0,0.0,0.0,89.22899483,50736.01705832,49922.01593,0.0,1147.89187382,0.0,89.42888505,0.0,0.0,0.0,0.0,100000.0,0.0,160.45091705,381.78959028,582.10976515,77980.77728507,66007.68592468:0.0
1.04237,1.04694,1.05083,1.05083,1.05083,1.05083,1.03836,1.03836,1.03976,1.038,1.03,1.03354,1.03594,1.03594,1.03822,1.03688,1.04524,1.05415,1.06209,1.06365,1.06901,1.0619,1.03719,1.03407:0.0,1006.35787227,785.0,0.0,0.0,0.0,389.08880978,0.0,786.52672732,17637.57891496,50.0,3354.07669342,14493.31406847,0.0,1317.30366207,30.141167,4811.56932335,2583.59443535,3121.11556879,1250.84389849,32865.52501347,31.4760809,8787.16722751,7621.63717292:0.984
0.81029,0.80915,0.80915,0.80915,0.80973,0.81402,0.81591,0.81591,0.81117,0.81161,0.8079,0.80755,0.80657,0.80764,0.7902,0.7922,0.79455,0.78,0.78214,0.7882,0.78138,0.7795,0.76734,0.75493:5580.19171348,2644.18523116,0.0,0.0,2586.36995265,9237.65020029,85.53697098,0.0,1329.5539397,169.2355072,7267.97461591,1505.0541243,5490.55524868,2569.65308769,29880.45092404,4875.36680243,217.178064,60973.67142221,6562.2861599,5671.99469217,7797.34275297,51970.62309491,31759.70687449,147864.39031752:0.247
0.55468,0.56024,0.55795,0.55616,0.5524,0.5524,0.5524,0.5524,0.555,0.5677,0.56587,0.57028,0.5578,0.55389,0.55384,0.55996,0.55699,0.5635,0.55437,0.56142,0.56581,0.56508,0.55754,0.55733:2227.26042974,791.15291024,864.11413839,100208.73493639,67.15831164,0.0,0.0,0.0,50.0,8841.35676067,9585.5326644,15087.13710294,5088.80866263,12075.727461,115140.78786232,1033.48206155,11526.43269601,5711.33250722,25502.58395814,1351.83990172,22829.51021757,2431.66911771,7470.2031079,417.39890448:0.3084
0.4912,0.50082,0.52662,0.55125,0.5454,0.58156,0.58762,0.56402,0.57494,0.57076,0.58038,0.58121,0.57216,0.5805,0.57616,0.56847,0.55066,0.55807,0.54694,0.55021,0.54737,0.5562,0.53847,0.53342:35731.36893812,170377.27602947,45803.87395035,18035.1591561,20750.65294288,149291.94169274,95408.57603559,137502.09480532,27620.51358042,47539.56303284,55359.38797536,26361.21816412,19553.32805887,25235.87850142,84275.70074753,204754.56679949,5858.28435788,3754.68149529,9451.19007703,4428.67984731,46212.94914326,66470.9422452,23092.08957902,2101.17730868:0.3633
1.21568,1.20968,1.20968,1.20968,1.20938,1.20938,1.20172,1.20527,1.20387,1.20387,1.20181,1.20691,1.21704,1.20102,1.18734,1.18731,1.18731,1.18731,1.18731,1.19773,1.18819,1.18819,1.19096,1.17142:27676.04537999,42.20513254,0.0,0.0,9257.33644644,0.0,444.0,246.98078064,661.21749026,0.0,63.98424281,3524.50584119,3962.52409772,22859.00381054,15125.47536323,4308.53476447,0.0,0.0,0.0,8831.03145702,3861.42705498,0.0,56.703179,37527.18446021:0.1193
0.45881,0.45658,0.45479,0.45258,0.45054,0.45352,0.45039,0.45039,0.4535,0.45588,0.4556,0.4556,0.45,0.44011,0.4443,0.44558,0.44669,0.44853,0.44639,0.44702,0.44562,0.44001,0.435,0.43511:14131.75924174,1420.21497435,1877.05454656,33482.05076628,1620.18837873,3979.20235485,662.77550567,0.0,1103.0,10129.5,779.28253962,0.0,75164.30813353,38028.80495774,7037.88686434,59775.25,704.38467417,2790.0,8371.68324082,2006.0,7548.45324817,7484.09854995,117036.87735884,5954.06791191:0.0168
0.62506,0.62506,0.62506,0.62913,0.62619,0.62619,0.62308,0.62308,0.61758,0.61007,0.6108,0.61397,0.59659,0.58287,0.58966,0.5872,0.57474,0.58357,0.5784,0.5747,0.57959,0.58185,0.58185,0.57453:710.16539212,0.0,0.0,42043.88628879,35696.86868099,0.0,2933.0,2238.19658044,17234.5869941,23922.73497722,41740.33542526,6379.98174519,111272.74724555,85867.98634555,23878.93008769,29069.4639378,42432.88601569,19001.77087429,22740.09719995,45294.835502,13702.09659298,3466.75416886,0.0,8120.37388148:-0.0935
0.77893,0.78237,0.79004,0.7888,0.78499,0.79157,0.7915,0.7915,0.79191,0.79505,0.78513,0.77505,0.78984,0.76999,0.78462,0.78081,0.77989,0.78106,0.78106,0.7825,0.7825,0.78669,0.78669,0.78669:5083.2776621,12.71808576,4513.39607,2461.14309968,4365.62455365,10205.23363182,403.30942997,0.0,857.80993852,4015.3213435,23154.28724201,1788.45324777,2730.28967892,3000.05807609,3304.64547128,3440.02417075,34018.69903094,11160.91922137,0.0,2181.47057025,0.0,5152.87786978,0.0,0.0:-0.0258
1.09059,1.08582,1.09324,1.09335,1.09205,1.09744,1.08777,1.08777,1.09439,1.093,1.08621,1.07,1.06834,1.05923,1.08046,1.08691,1.069,1.06126,1.06126,1.04697,1.05006,1.03934,1.03047,1.02305:0.0,7353.06811237,3265.65561904,815.83284086,10000.0,1007.20417513,499.14994188,0.0,9771.08819923,167.949813,1554.42500958,6713.9807064,3418.06258405,11877.90374654,1045.47433834,31.574453,102.3663236,203000.0,7630.82362219,3005.74826339,101441.78728394,143.32466777,2216.23032368,104913.19105011:0.0549
0.52543,0.51538,0.51715,0.529,0.5325,0.53423,0.52864,0.52598,0.52986,0.53392,0.54404,0.52392,0.51803,0.52535,0.53479,0.53129,0.52307,0.51462,0.52479,0.5222,0.5273,0.5273,0.53318,0.53318:866.91722631,1005.68490724,4751.99450042,1604.25287356,38620.85948973,9228.14849242,46279.3785185,24313.91043536,14403.73426936,47997.05222309,42589.84222166,13474.76761202,22950.27137562,91029.97894186,5592.24629099,11120.67086353,7567.7754105,7531.32322482,221.09171306,3024.80472931,70.96633832,0.0,76.47566487,0.0:0.8135
0.4566,0.4566,0.46019,0.45894,0.4579,0.4587,0.46138,0.46096,0.45879,0.45894,0.45894,0.46066,0.45685,0.45837,0.4546,0.45192,0.45192,0.45101,0.44157,0.43914,0.43983,0.44927,0.45038,0.45012:2184.09354846,0.0,2774.06294336,32.55379788,1528.05857578,3288.0,1838.11347816,542.0,20508.23428014,7355.0,0.0,50000.0,46122.83011071,1843.90350596,579.04962152,32933.8003886,0.0,14828.28587117,20203.26694095,4155.89896377,12137.34913962,51711.30032318,637.28925213,2982.07709629:0.0
0.54125,0.53896,0.5395,0.54281,0.542,0.54617,0.562,0.55875,0.55487,0.55923,0.55773,0.55549,0.55327,0.56202,0.56238,0.56095,0.55595,0.5581,0.55653,0.55838,0.56217,0.56468,0.56099,0.56099:0.0,4246.0,32292.73623127,80.88172657,1000.0,6284.97063903,249616.46888129,10896.68776014,3705.30194146,6645.36843437,659.49711271,158040.38390539,253.04528592,2522.14733753,104278.29017244,4310.912866,2656.64325829,1725.47182335,786.25538309,45364.57794668,16096.46707133,5114.38209128,3565.1122852,0.0:0.3487
