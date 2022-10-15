
#include <iostream>
#include <memory>
#include <string>

#include "argparse/argparse.hpp"

#include "moab/Core.hpp"

#include "gprt.h"

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

    std::shared_ptr<moab::Core> mbi = std::make_shared<moab::Core>();

    std::cout << "Loading " << filename << "..." << std::endl;
    mbi->load_file(filename.c_str());

    GPRTContext gprt = gprtContextCreate(nullptr, 1);


    return 0;
}