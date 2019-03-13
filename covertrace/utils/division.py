from covertrace.data_array import Sites, DataArray
import numpy as np

def get_id_dic(id_array):
    # given larr with all cell_ids, returns dictionary with key = cell_id and value = index of cell_id in larr
    id_dic = {}
    for n, ids in enumerate(id_array):
        cell = int(np.unique(np.array(ids[~np.isnan(ids)])))
        id_dic[cell] = n
        
    return id_dic

def connect_parent_daughter(sites, cell_label='nuclei', channel='TRITC'):
    #goes in order through the set of cells
    #concatenates parent trace to the front of the daughter trace, at the frame where division is detected
    
    for pos, larr in sites.iteritems():
        print pos, larr.shape
        labels = larr.labels
        id_index = get_id_dic(larr[cell_label, channel, 'cell_id'])
        to_blank = []
        
        for child_idx, l in enumerate(larr[cell_label, channel, 'parent']):
            #print child_idx
            division_frame = np.where(~np.isnan(l)) 
            if len(division_frame[0]) > 0:
                division_frame = int(division_frame[0])
                parent = int(l[division_frame])
                
                #print division_frame, parent
                #print parent
                
                try:
                    parent_idx = id_index[parent] #uses generated dictionary of cells and indicies
                    
                    parent_data = larr[:, parent_idx, :][:,:division_frame]
                    child_data = larr[:, child_idx, :][:,division_frame:]
                    #print parent_data.shape
                    #print child_data.shape
                    concat = DataArray(np.column_stack((parent_data, child_data)), labels)
                    #print concat.shape

                    larr[:, child_idx, :] = concat
                    to_blank.append(parent_idx)

                except KeyError:
                    print 'parent for cell at index ' + str(child_idx) + ' not found in set'
            
            

        blank_data = np.empty(concat.shape)
        blank_data[:] = np.nan
        blank_array = DataArray(blank_data, labels)
            
        for par in to_blank:
            larr[:, par, :] = blank_array 
        
        print str(len(to_blank)) + ' parents connected to daughter cells'