#include <iostream>
#include <fstream>
using namespace std;
int main(){
    fstream fs;
    fs.open("test.csv", ios::in);
    fs << "test" << "," << "test2";
    fs.close();
}