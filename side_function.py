from pyproj import Proj, transform
import shapefile
from math import sin, cos, sqrt, atan2, radians
from datetime import timedelta
import gdal
import numpy as np
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def covert_coordinate_from_4326_to_MA(point):
    y1, x1 = point[0], point[1]
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:26986')
    x2, y2 = transform(inProj, outProj, x1, y1)
    return (y2, x2)

def maintenance_depots(shpname):
    m_d=[]
    sf=shapefile.Reader(shpname)
    shp=sf.shapes()
    length=len(shp)
    # print("\nMaintenance depots: "+str(length))
    for s in shp:
        point=s.points[0]
        output_point=(point[1],point[0])
        m_d.append(output_point)
    return m_d

def find_furthest_point(point,point_set,included_set,r):
    distance=0
    p_f=(0,0)
    for p in point_set:
        if p not in included_set:
            distance_n = cal_distance_simplified(p, point)
            if distance_n > distance:
                distance = distance_n
                p_f = p
    if distance>r:
        return p_f
    return None


def cal_distance_simplified(point_1,point_2):
    return (sqrt((point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2))



def find_minimum_time_gap(start_time_set,duration_set,severity,subtype,excluded_set,original_point_set):

    index_set=[]
    for p in excluded_set:
        index_set.append(original_point_set.index(p))
    indexed_incident=[[start_time_set[i],start_time_set[i]+timedelta(minutes=duration_set[i])]for i in index_set]
    for i in range(len(excluded_set)):
        indexed_incident[i]+=[excluded_set[i],severity[i],subtype[i]]
    indexed_incident.sort(key=lambda index_time: index_time[0])
    if len(indexed_incident)==1:
        return 1e8,1e8,1,[],[]

    start_time=[t[0] for t in indexed_incident]
    end_time=[t[1] for t in indexed_incident]
    location=[t[2] for t in indexed_incident]
    severity=[t[3] for t in indexed_incident]
    subtype=[t[4] for t in indexed_incident]

    same_time_threshold=25
    gap_set=[]
    min_gap = timedelta(1e8)
    current_end_time=end_time[0]
    current_occuring=[(start_time[0],end_time[0]+timedelta(minutes=same_time_threshold),location[0],severity[0],subtype[0])]
    correlated_incident_set=[]
    parallel_incident_set=[]

    for i in range(1,len(indexed_incident)):
        # gap
        gap=max(timedelta(0),start_time[i]-current_end_time)
        gap_set.append(gap)
        if gap<min_gap:
            min_gap=gap
        current_end_time=max(current_end_time,end_time[i])
        # same time incident
        current_occuring = [elem for elem in current_occuring if elem[1]>start_time[i]]
        # same_time_incident=max(same_time_incident,len(current_occuring)+1)
        current_occuring.append((start_time[i],end_time[i]+timedelta(minutes=same_time_threshold),
                                 location[i],severity[i],subtype[i]))
        if len(current_occuring)>1:
            correlated_incidents,parrallel_incidents=find_correlated(current_occuring)

            for full_info in correlated_incidents:

                c_i=full_info
                # c_i=[f[2] for f in full_info]
                if len(correlated_incident_set)==0:
                    correlated_incident_set.append(c_i)
                    continue
                if c_i[0] in correlated_incident_set[-1]:
                    correlated_incident_set.pop(-1)
                correlated_incident_set.append(c_i)

            for p_i in parrallel_incidents:
                if len(parallel_incident_set)==0:
                    parallel_incident_set.append(p_i)
                    continue
                if list(p_i.keys())[0] in list(parallel_incident_set[-1].keys()):
                    parallel_incident_set.pop(-1)
                parallel_incident_set.append(p_i)

    td_mins = round(min_gap.total_seconds() / 60, 1)
    if len(gap_set)>1:
        td_average=np.mean([round(gap.total_seconds() / 60, 1) for gap in gap_set])
    else:
        td_average=round(gap_set[0].total_seconds() / 60, 1)

    try:
        same_time_incident=max([len(p_i) for p_i in parallel_incident_set])
    except:
        same_time_incident = 1

    return td_mins,td_average,same_time_incident,correlated_incident_set,parallel_incident_set


def find_correlated(current_occuring):
    loc=[incident[2] for incident in current_occuring]
    n=len(loc)
    distance_threshold=100
    correlated_incidents=[]
    parallel_incidents=[]

    center=[]
    group={}
    for i in range(n):
        Independence=True
        for c in center:
            if cal_distance_simplified(loc[i],c)<distance_threshold:
                group[c].append(current_occuring[i])
                Independence=False
        if Independence:
           center.append(loc[i])
           group[loc[i]] = [current_occuring[i]]
    for g in group:
        if len(group[g])>1:
            correlated_incidents.append(group[g])
    if len(group)>1:
        parallel_incidents.append(group)
    return correlated_incidents, parallel_incidents

def MIC_analyses(MIC_full_info):
    type_sequence = []
    type_sequence_num = []
    severity = []
    start_time_gap_population = []
    record=[]
    for MIC in MIC_full_info:
        start_time = [i[0] for i in MIC]
        start_time_gap = [round((start_time[i + 1] - start_time[i]).total_seconds() / 60, 1)
                          for i in range(len(start_time) - 1)]
        start_time_gap_population += start_time_gap
        min_start_time = min(start_time)
        end_time = max(i[1] for i in MIC) - timedelta(minutes=25)
        duration = round((end_time - min_start_time).total_seconds() / 60, 1)
        types = [i[4] for i in MIC]
        if types not in type_sequence:
            type_sequence.append(types)
            type_sequence_num.append(1)
        else:
            type_sequence_num[type_sequence.index(types)] += 1
        s = max([int(list((filter(str.isdigit, i[3])))[0]) for i in MIC])
        severity.append((len(types), s, duration))
        record.append((MIC[0][2][0],MIC[0][2][1],len(types),s))
    record.sort(key=lambda ele:int(ele[3]))
    flink = open('MIC.txt','w+')
    for r in record:
        flink.write(
            '%s,%s,%s,%s\n' % (r[0],r[1],r[2],r[3]))
    flink.close()

    subsequence = [(type_sequence[i], type_sequence_num[i], len(type_sequence[i])) for i in range(len(type_sequence))]
    subsequence.sort(key=lambda k: k[1], reverse=True)
    severity_and_duration(severity)


def severity_and_duration(severity):
    severity_dic={}
    for i in severity:
        num_of_inc=i[0]
        if num_of_inc in severity_dic.keys():
            severity_dic[num_of_inc].append((i[1],i[2]))
        else:
            severity_dic[num_of_inc]=[(i[1],i[2])]
    num_of_level2=[sum([1 for i in severity_dic[2] if i[0] == 2]),
     sum([1 for i in severity_dic[3] if i[0] == 2]),
     sum([1 for i in severity_dic[4] if i[0] == 2])]
    num_of_level3 =[sum([1 for i in severity_dic[2] if i[0] == 3]),
     sum([1 for i in severity_dic[3] if i[0] == 3]),
     sum([1 for i in severity_dic[4] if i[0] == 3])]
    mean_of_duration=[np.mean([i[1] for i in severity_dic[2]]),
     np.mean([i[1] for i in severity_dic[3]]),
     np.mean([i[1] for i in severity_dic[4]])]


def circle_rank(center_set,point_set,r):
    circle_info=[]
    max_cov=0
    for center in center_set:
        included_set=find_included(center,point_set,[],r)
        if len(included_set)>max_cov:
            circle_info=(center,included_set)
            max_cov=len(included_set)
    return circle_info


def find_included(point, point_set, included_set,distance):
    for p in point_set:
        if p not in included_set:
            distance_n=cal_distance_simplified(p, point)
            if distance_n<=distance:
                included_set.append(p)
    return included_set



def read_raster(raster_name):
    ds = gdal.Open(raster_name)
    cols=ds.RasterXSize
    rows=ds.RasterYSize
    geo= ds.GetGeoTransform()
    originX=geo[0]
    originY=geo[3]
    pixelWidth=geo[1]
    pixelHeight=geo[5]
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows)
    point_set=[]
    for c in range(cols):
        for r in range(rows):
            if data[r,c]==1:
                point=cal_coor_from_offset(c,r,originX,originY,pixelWidth,pixelHeight)
                point_set.append(point)
    return point_set



def cal_coor_from_offset(xoffset,yoffset,originX,originY,pixelWidth,pixelHeight):
    x = xoffset * pixelWidth + originX
    y = yoffset * pixelHeight + originY
    return (y,x)



def cal_cost(same_time_num,r):
    r_in_mile=r / 1000 / 1.60934

    cost_linear=0
    alpha=0.5
    b=1
    individual_cost_linear=r_in_mile*alpha+b

    cost_exponential=0
    base=1.25
    individual_cost_exponential=base**r_in_mile

    for num in same_time_num:
        cost_linear+=num*individual_cost_linear
        cost_exponential+=num*individual_cost_exponential

    return (round(cost_linear,1),round(cost_exponential,1))


def raster_coverage(center_point_set,key_station,point_set,name,r):
    included_set=[]
    key_coverage=0
    for c_p in center_point_set:
        old_included_set=included_set.copy()
        included_set=find_included(c_p[0],point_set,included_set,r)
        if c_p in key_station:
            key_coverage+=len(included_set)-len(old_included_set)
    coverage=len(included_set)/len(point_set)
    k_c=key_coverage/len(point_set)
    print('%s converage:'%name,coverage,' key coverage:',k_c)




def generate_random_circle(point_set,n,r):
    center_set=[]
    for point in point_set:
        for i in range(n):
            center_x=point[0]+random.uniform(-1,1)*r/100
            center_y=point[1]+random.uniform(-1,1)*r/100
            center_set.append((center_x,center_y))
    return center_set

def sortSecondLen(val):
    return len(val[1])

def draw(point_set,center_set,r,frequency=None):
    fig = plt.figure(figsize=(13, 6), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    scatter1=[p[0] for p in point_set]
    scatter2=[p[1] for p in point_set]
    plt.scatter(scatter2,scatter1,s=1,edgecolors='k')
    for i in range(len(center_set)):
        p=center_set[i]
        f=frequency[i]
        if frequency is not None:
            c,a=decide_color(f,threshold=120)
        patch = patches.Circle((p[1],p[0]), radius=r,fc=c,alpha=a)
        ax.add_patch(patch)
    plt.legend((patches.Circle((0,0), radius=r,fc='r',alpha=0.2),patches.Circle((0,0), radius=r,fc='g',alpha=0.1)), ('Key stations', 'Regular Stations'))
    plt.savefig('station.png')

def decide_color(f,threshold):
    if f>=threshold:
        return 'r',0.2
    else:
        return 'g',0.1

def get_between(x, d):
    for i in range(len(x)):
        if d<=x[i+1]:
            return i

def greedy_generate(point_set,r):
    original_point_set=point_set.copy()
    final_center_set=[]
    while len(point_set)>len(original_point_set)*0.3:
        print(len(point_set))
        center_set = generate_random_circle(point_set, 1,r)
        circle_info=circle_rank(center_set,point_set)
        final_center_set.append(circle_info[0])
        excluded_set=circle_info[1]
        excluded_set.sort(reverse=True)
        for point in excluded_set:
            point_set.pop(point)
    print(len(final_center_set))
    draw(original_point_set,final_center_set)

def draw_figures(min_gap,key_min,correlated_num,parallel_num,same_time_num,r):
    start = 0
    end = 125
    bin = int((end - start) / 5)
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(np.append(np.arange(start, end, bin), [1e8]), a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(start + bin / 2, end, bin), labels, width=bin)
    plt.xticks(np.arange(start + bin / 2, end, bin),
               ['%s - %s' % (start, start + bin), '%s - %s' % (start + bin, start + bin * 2),
                '%s - %s' % (start + bin * 2, start + bin * 3), '%s - %s' % (start + bin * 3, start + bin * 4),
                'more than\n %s' % (start + bin * 4)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0, end)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + bin / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('All stations', fontsize=16)
    plt.xlabel('minimum time gap (min)', fontsize=16)
    plt.savefig('min time gap.png')
    plt.close()

    min_gap = key_min
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(np.append(np.arange(start, end, bin), [1e8]), a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(start + bin / 2, end, bin), labels, width=bin)
    plt.xticks(np.arange(start + bin / 2, end, bin),
               ['%s - %s' % (start, start + bin), '%s - %s' % (start + bin, start + bin * 2),
                '%s - %s' % (start + bin * 2, start + bin * 3), '%s - %s' % (start + bin * 3, start + bin * 4),
                'more than\n %s' % (start + bin * 4)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0, end)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + bin / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Key stations', fontsize=16)
    plt.xlabel('minimum time gap (min)', fontsize=16)
    plt.savefig('key min time gap.png')
    plt.close()

    start = 1
    end = 6
    bin = int((end - start + 1) / 3)
    hist_range = np.append([0], np.append(np.arange(start - 0.5, end + 1, bin), [1e8]))
    min_gap = correlated_num
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of correlated incident groups', fontsize=16)
    plt.savefig('correlated incident.png')
    plt.close()

    min_gap = parallel_num
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of parallel incident groups', fontsize=16)
    plt.savefig('parallel incident.png')
    plt.close()

    min_gap = [correlated_num[i] + parallel_num[i] for i in range(len(correlated_num))]
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of incident groups\n ocurring at the same time at each station', fontsize=16)
    plt.savefig('total incident.png')
    plt.close()

    same_time_incident_groups = [0 for i in range(4)]
    for a_g in same_time_num:
        d = get_between(np.arange(0.5, 5, 1), a_g)
        same_time_incident_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)

    labels = [f / len(same_time_num) for f in same_time_incident_groups]
    labels = same_time_incident_groups

    rects = ax.bar(np.arange(1, 5, 1), labels, width=1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 4.5)
    # axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        # ax.text(get_x + 1/2, height + y_limit[1]*0.04, "{0:.1f}%".format(labels[i] * 100), ha='center', va='center',fontsize=18)
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04, labels[i], ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Highest number of parallel incidents\n occurring at the same time', fontsize=16)
    plt.savefig('same time incident.png')
    plt.close()


def cal_distance(point_1,point_2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(point_1[0])
    lon1 = radians(point_1[1])
    lat2 = radians(point_2[0])
    lon2 = radians(point_2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance