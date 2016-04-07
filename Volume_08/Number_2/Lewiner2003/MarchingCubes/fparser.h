/***************************************************************************\
|* Function parser v2.62 by Warp                                           *|
|* http://www.students.tut.fi/~warp/FunctionParser/                        *|
|* -----------------------------                                           *|
|* Parses and evaluates the given function with the given variable values. *|
|*                                                                         *|
\***************************************************************************/

#ifndef ONCE_FPARSER_H_
#define ONCE_FPARSER_H_

#if defined(WIN32) && !defined(__CYGWIN__)
#pragma warning(disable:4786)
#endif // WIN32

#include <string>
#include <map>
#include <vector>
#include <iostream>

class FunctionParser
{
public:
    enum ParseErrorType
    {
        NO_SYNTAX_ERROR=-1,
        SYNTAX_ERROR=0, MISM_PARENTH, MISSING_PARENTH, EMPTY_PARENTH,
        EXPECT_OPERATOR, OUT_OF_MEMORY, UNEXPECTED_ERROR, INVALID_VARS,
        ILL_PARAMS_AMOUNT, PREMATURE_EOS, EXPECT_PARENTH_FUNC
    };


    int Parse(const std::string& Function, const std::string& Vars,
              bool useDegrees = false);
    const char* ErrorMsg() const;
    inline ParseErrorType GetParseErrorType() const { return parseErrorType; }

    float Eval(const float* Vars);
    inline int EvalError() const { return evalErrorType; }

    bool AddConstant(const std::string& name, float value);

    typedef float (*FunctionPtr)(const float*);

    bool AddFunction(const std::string& name,
                     FunctionPtr, unsigned paramsAmount);
    bool AddFunction(const std::string& name, FunctionParser&);

    void Optimize();


    FunctionParser();
    ~FunctionParser();

    // Copy constructor and assignment operator (implemented using the
    // copy-on-write technique for efficiency):
    FunctionParser(const FunctionParser&);
    FunctionParser& operator=(const FunctionParser&);


    // For debugging purposes only:
    void PrintByteCode(std::ostream& dest) const;

public:
  bool *using_var;



//========================================================================
private:
//========================================================================

// Private data:
// ------------
    ParseErrorType parseErrorType;
    int evalErrorType;

    struct Data
    {
        unsigned referenceCounter;

        int varAmount;
        bool useDegreeConversion;

        typedef std::map<std::string, unsigned> VarMap_t;
        VarMap_t Variables;

        typedef std::map<std::string, float> ConstMap_t;
        ConstMap_t Constants;

        VarMap_t FuncPtrNames;
        struct FuncPtrData
        {
            FunctionPtr ptr; unsigned params;
            FuncPtrData(FunctionPtr p, unsigned par): ptr(p), params(par) {}
        };
        std::vector<FuncPtrData> FuncPtrs;

        VarMap_t FuncParserNames;
        std::vector<FunctionParser*> FuncParsers;

        unsigned* ByteCode;
        unsigned ByteCodeSize;
        float* Immed;
        unsigned ImmedSize;
        float* Stack;
        unsigned StackSize;

        Data();
        ~Data();
        Data(const Data&);

        Data& operator=(const Data&); // not implemented on purpose
    };

    Data* data;

    // Temp data needed in Compile():
    unsigned StackPtr;
    std::vector<unsigned>* tempByteCode;
    std::vector<float>* tempImmed;


// Private methods:
// ---------------
    inline void copyOnWrite();


    bool checkRecursiveLinking(const FunctionParser*) const;

    bool isValidName(const std::string&) const;
    Data::VarMap_t::const_iterator FindVariable(const char*,
                                                const Data::VarMap_t&) const;
    Data::ConstMap_t::const_iterator FindConstant(const char*) const;
    int CheckSyntax(const char*);
    bool Compile(const char*);
    bool IsVariable(int);
    void AddCompiledByte(unsigned);
    void AddImmediate(float);
    void AddFunctionOpcode(unsigned);
    inline void incStackPtr();
    int CompileIf(const char*, int);
    int CompileFunctionParams(const char*, int, unsigned);
    int CompileElement(const char*, int);
    int CompilePow(const char*, int);
    int CompileMult(const char*, int);
    int CompileAddition(const char*, int);
    int CompileComparison(const char*, int);
    int CompileAnd(const char*, int);
    int CompileOr(const char*, int);
    int CompileExpression(const char*, int, bool=false);


    void MakeTree(void*) const;
};

#endif
