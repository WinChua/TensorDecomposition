#include<algorithm>
#include<iostream>
#include<set>
#include<stdlib.h>
#include<stdio.h>
#include<string>
#include<sstream>
#include<map>
#include<fstream>
#include "lbfgs.h"


typedef std::map<int, std::map<int, std::map<int, float> > > TensorData;

class Tensor {
public:
    int In, Jn, Kn, L;
    std::string filename;
    Tensor(std::string f) {

        std::ifstream in(f.c_str());
        if (! in.is_open()) {
            printf("Error opening file\n");
            exit(1);
        }
        std::string line;
        int Iv, Jv, Kv;
        float V;
        while (std::getline(in, line)) {
            std::stringstream ss(line);
            ss >> Iv >> Jv >> Kv >> V;
            Iset.insert(Iv);
            Jset.insert(Jv);
            Kset.insert(Kv);
            data[Iv][Jv][Kv] = V;

        }
        in.close();
        In = *std::max_element(Iset.begin(), Iset.end()) + 1;
        Jn = *std::max_element(Jset.begin(), Jset.end()) + 1;
        Kn = *std::max_element(Kset.begin(), Kset.end()) + 1;
    }

    float get(int i, int j, int k) {
        if(contain(i, j, k)) {
            return data[i][j][k];
        }
        else {
            return 0.0;
        }
    }
    
    bool contain(int i, int j, int k) {
        TensorData::iterator it_find = data.find(i);
        if(it_find == data.end()) {
            return false;
        }
        else {
            std::map<int, std::map<int, float> >::iterator rt_find = (it_find->second).find(j);
            if(rt_find == (it_find->second).end()) {
                return false;
            }
            else{
                std::map<int, float>::iterator vt_find = (rt_find->second).find(k);
                if(vt_find == (rt_find->second).end()){
                    return false;
                }
                else {
                    return true;
                }
            }
        }
    }
    friend std::ostream& operator<<(std::ostream & out, const Tensor & d) {
        std::cout << "The first dimension is " << d.In << std::endl;
        std::cout << "The second dimension is " << d.Jn << std::endl;
        std::cout << "The third dimension is " << d.Kn << std::endl;

        for(TensorData::const_iterator it = d.data.begin(); it != d.data.end(); it++) {
            for(std::map<int, std::map<int, float> >::const_iterator rit = (it->second).begin(); 
                rit != (it->second).end(); rit++) {
                for(std::map<int, float>::const_iterator vit = (rit->second).begin();
                    vit != (rit->second).end(); vit++) {
                    out << it->first <<"," <<  rit->first << "," << vit->first << "," << vit->second << std::endl; 
                }
            }
        }
        return out;
    }
private:
    std::set<int> Iset, Jset, Kset;
    std::map<int, std::map<int, std::map<int, float> > > data;
};

    
