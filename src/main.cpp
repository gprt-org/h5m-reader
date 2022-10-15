
#include <iostream>
#include <string>


#include "argparse/argparse.hpp"

int main(int argc, char** argv) {

    argparse::ArgumentParser args("GPRT H5M READER");

    args.add_argument("filename");

    try {
    args.parse_args(argc, argv);                  // Example: ./main -abc 1.95 2.47
    }
    catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(0);
    }

    auto filename = args.get<std::string>("filename");

    std::cout << "Filename is: " << filename << std::endl;

    return 0;
}