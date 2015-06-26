/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
*/

#pragma once
#include <stdexcept>
#include <sstream>

#ifndef _NOEXCEPT
// _NOEXCEPT is required on Mac OS
// making sure other platforms don't barf
#define _NOEXCEPT throw () 
#endif

namespace VW {

	class vw_exception : public std::exception
	{
	private:
		// source file exception was thrown
		const char* file;

		std::string message;

		// line number exception was thrown
		int lineNumber;
	public:
		vw_exception(const char* file, int lineNumber, std::string message);

		vw_exception(const vw_exception& ex);

		~vw_exception() _NOEXCEPT;

		virtual const char* what() const _NOEXCEPT;

		const char* Filename() const;

		int LineNumber() const;
	};

// ease error handling and also log filename and line number
#define THROW(args) \
	{ \
		std::stringstream __msg; \
		__msg << args << std::endl; \
		throw VW::vw_exception(__FILE__, __LINE__, __msg.str()); \
	}

}
