#include <flecsi/utils/common.hh>
#include <flecsi/utils/factory.hh>
#include <flecsi/utils/ftest.hh>

class io_base_t
{
public:
  io_base_t(std::string & filename) : filename_(filename) {}

  virtual int32_t read() = 0;

  virtual ~io_base_t() {}

protected:
  std::string filename_;

}; // struct io_base_t

using io_factory_t =
  flecsi::utils::Factory_<io_base_t, std::string, std::string &>;

struct test_io_t : public io_base_t {

  test_io_t(std::string & filename) : io_base_t(filename) {}

  int32_t read() override {
    return 0;
  } // read

}; // struct test_io_t

io_base_t *
create_test_io(std::string & filename) {
  return new test_io_t(filename);
} // create_test_io

bool test_registered =
  io_factory_t::instance().registerType("tst", create_test_io);

// Creation handlers for the Factory_ type variants in TEST() below. These
// accept the arguments as given in those types, and return pointers to the
// return types as given. Keys play no role here.

float *
a_foo(int, float, double) {
  return new float(1.2f);
}

float *
b_foo(int, float, double) {
  return new float(3.4f);
}

double *
bar_1(double, double) {
  return new double(5.6);
}

double *
bar_2(double, double) {
  return new double(8.9);
}

// These handlers do something slightly more interesting. Well, less boring.

float *
add_ifd(int i, float f, double d) {
  return new float(i + f + float(d));
}

double *
add_dd(double d1, double d2) {
  return new double(d1 + d2);
}

int
factory(int, char **) {
  FTEST {
    // ------------------------
    // types
    // ------------------------

    // one variant
    using factory_charkey_t = // return, key,  arguments...
      flecsi::utils::Factory_<float, char, int, float, double>;

    FTEST_CAPTURE() << FTEST_TTYPE(factory_charkey_t::createHandler)
                    << std::endl;
    FTEST_CAPTURE() << FTEST_TTYPE(factory_charkey_t::key_t) << std::endl;
    FTEST_CAPTURE() << FTEST_TTYPE(factory_charkey_t::map_t) << std::endl;
    FTEST_CAPTURE() << std::endl;

    // another variant
    using factory_longkey_t = // return, key,  arguments...
      flecsi::utils::Factory_<double, long, double, double>;

    FTEST_CAPTURE() << FTEST_TTYPE(factory_longkey_t::createHandler)
                    << std::endl;
    FTEST_CAPTURE() << FTEST_TTYPE(factory_longkey_t::key_t) << std::endl;
    FTEST_CAPTURE() << FTEST_TTYPE(factory_longkey_t::map_t) << std::endl;
    FTEST_CAPTURE() << std::endl;

    // ------------------------
    // instance
    // ------------------------

    // Note that the instance() returns can be referenced only,
    // because Factory_'s copy c'tor was (rightfully) deleted.

    // of the first variant above
    const factory_charkey_t & facchar1 = factory_charkey_t::instance();
    const factory_charkey_t & facchar2 = factory_charkey_t::instance();
    EXPECT_EQ(&facchar1, &facchar2); // because singleton

    // of the second variant above
    const factory_longkey_t & faclong1 = factory_longkey_t::instance();
    const factory_longkey_t & faclong2 = factory_longkey_t::instance();
    EXPECT_EQ(&faclong1, &faclong2); // because singleton

    // dumb check, but why not...
    EXPECT_NE((void *)&facchar1, (void *)&faclong1);

    // ------------------------
    // registerType
    // ------------------------

    // For illustration, we'll use both the above-created instances,
    // as well as on-the-fly instances.

    bool a = facchar1.instance().registerType('a', a_foo);
    bool b = facchar2.instance().registerType('b', b_foo);
    // 'c' can associate with b_foo, even though something else does too...
    bool c = factory_charkey_t::instance().registerType('c', b_foo);
    // 'b' can't re-associate...
    bool b2 = factory_charkey_t::instance().registerType('b', b_foo);

    FTEST_CAPTURE() << int(a) << std::endl; // true
    FTEST_CAPTURE() << int(b) << std::endl; // true
    FTEST_CAPTURE() << int(c) << std::endl; // true
    FTEST_CAPTURE() << int(b2) << std::endl; // false ('b' didn't re-associate)
    FTEST_CAPTURE() << std::endl;

    bool bar1 = faclong1.instance().registerType(1, bar_1);
    bool bar2 = faclong1.instance().registerType(2, bar_2);
    bool bar3 = faclong1.instance().registerType(3, bar_1); // ok, same handler
    bool bar4 = faclong1.instance().registerType(2, bar_2); // bad, same key

    FTEST_CAPTURE() << int(bar1) << std::endl;
    FTEST_CAPTURE() << int(bar2) << std::endl;
    FTEST_CAPTURE() << int(bar3) << std::endl;
    FTEST_CAPTURE() << int(bar4) << std::endl;
    FTEST_CAPTURE() << std::endl;

    // ------------------------
    // create
    // ------------------------

    // Recall:
    //   'a'      ==> a_foo: return 1.2
    //   'b', 'c' ==> b_foo: return 3.4
    //    1, 3    ==> bar_1: return 5.6
    //    2       ==> bar_2: return 8.9
    // So, let's get those somewhat boring results, independent of arguments...

    float * f1 = factory_charkey_t::instance().create('a', 0, 0.f, 0.0);
    float * f2 = factory_charkey_t::instance().create('b', 0, 0.f, 0.0);
    float * f3 = factory_charkey_t::instance().create('c', 0, 0.f, 0.0);
    double * d1 = factory_longkey_t::instance().create(1, 1.0, 2.0);
    double * d2 = factory_longkey_t::instance().create(3, 3.0, 4.0);
    double * d3 = factory_longkey_t::instance().create(2, 5.0, 6.0);

    FTEST_CAPTURE() << *f1 << std::endl;
    FTEST_CAPTURE() << *f2 << std::endl;
    FTEST_CAPTURE() << *f3 << std::endl;
    FTEST_CAPTURE() << std::endl;

    FTEST_CAPTURE() << *d1 << std::endl;
    FTEST_CAPTURE() << *d2 << std::endl;
    FTEST_CAPTURE() << *d3 << std::endl;
    FTEST_CAPTURE() << std::endl;

    // Recall:
    //    add_ifd: return i + f + d
    //    add_dd:  return d1 + d2
    // Register those handles, and get some slightly less boring results...

    (void)factory_charkey_t::instance().registerType('+', add_ifd);
    (void)factory_longkey_t::instance().registerType(123, add_dd);

    float * f4 = factory_charkey_t::instance().create('+', 1, 2.3f, 4.5);
    double * d4 = factory_longkey_t::instance().create(123, 6.7, 8.9);

    FTEST_CAPTURE() << *f4 << std::endl;
    FTEST_CAPTURE() << *d4 << std::endl;

    // Cleanup
    delete d4;
    delete f4;
    delete d3;
    delete d2;
    delete d1;
    delete f3;
    delete f2;
    delete f1;

    // ------------------------
    // Compare
    // ------------------------
#ifdef __GNUG__
    EXPECT_TRUE(FTEST_EQUAL_BLESSED("factory.blessed.gnug"));
#elif defined(_MSC_VER)
    EXPECT_TRUE(FTEST_EQUAL_BLESSED("factory.blessed.msvc"));
#else
    EXPECT_TRUE(FTEST_EQUAL_BLESSED("factory.blessed"));
#endif
  };
}

ftest_register_driver(factory);
