#pragma once
#include<iostream>
#include<windows.h>
#include<string>
#include<sstream>
#include<vector>

using namespace std;

class __declspec(dllexport) CTimeString {

	int m_year, m_month, m_day, m_hour, m_minute, m_second, m_milliseconds;

	void setTime();

	void getTimeValueFromString(string string_, int &i_minute, int &i_second, int &i_millisecond);
	void getTimeValueFromString(string string_, int &i_hour, int &i_minute, int &i_second, int &i_millisecond);

public:

	//can use "std::to_string" for it ...but detail is unknown
	template < typename T > std::string to_string(const T& n)
	{
		std::ostringstream stm;
		stm << n;
		return stm.str();
	}


	string getTimeString();

	string getTimeElapsefrom2Strings(string s_former, string s_latter);	//output  second

	string getTElapsefrom2S(string s_former, string s_latter);

	std::vector<int> find_all(const std::string str, const std::string subStr);

};

/*

CTimeString time;

while (1) {
cout <<time.getTimeStirng()<< endl;
Sleep(0.5 * 1000);
if (toggle.isToggleChanged()) getchar();
}

*/
