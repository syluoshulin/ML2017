import numpy as np
import string
import sys
import matplotlib.pyplot as plt

train_path = sys.argv[1]
test_path = sys.argv[2]

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,_) = read_data(train_path,True)
    tag_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']

    ###
    train_tag = to_multi_categorical(Y_data,tag_list)
    statistics_train_tag = np.sum(train_tag,axis=0) 
    for i in range(38):
        print(tag_list[i], ':', statistics_train_tag[i])
    
    fig, ax = plt.subplots()
    plt.plot(statistics_train_tag)
    ax.set_xticks(np.arange(0, 37, 1))
    ax.set_xticklabels(tag_list,rotation=68,fontsize=11)
    ax.set_aspect(0.005)
    # plt.yticks(tag_list)
    #plt.yticklabels(tag_list)
    plt.xlabel('Tags',fontsize=12)
    plt.ylabel('Number',fontsize=12)
    plt.title('TAGS\' DISTRIBUTION',fontsize=14,fontweight='bold')
    plt.tight_layout(pad=1)
    plt.show()

if __name__=='__main__':
    main()
